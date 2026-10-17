# Tests del orquestador: se ejecutan SIN API key usando LLM falsos que siguen un guion
# Las herramientas, la validación, los guardarraíles y el grafo son los REALES; solo cambia el "cerebro" (el LLM)
# Cubren: delegación completa, estado, aislamiento de contexto, supervisor infinito, guardarraíles, validación y degradación honesta

# Importa asyncio para correr las funciones asíncronas dentro de tests normales
import asyncio
# Importa json para leer lo que devuelven las herramientas
import json
# Importa Any para tipar
from typing import Any

# Importa pytest para los tests y para esperar excepciones
import pytest
# Importa los tipos de mensajes
from langchain_core.messages import AIMessage, HumanMessage

# Importa la configuración, que también se prueba
from app import config
# Importa la base de reseñas para armar los guiones con datos reales
from app import knowledge_base
# Importa las herramientas y funciones del analista
from app.agents.analyst_agent import calcular_estadisticas, etiquetar, puntuar_sentimiento
# Importa la herramienta de búsqueda del investigador
from app.agents.research_agent import buscar_resenas
# Importa el grupo de LLM y el armado del grafo
from app.graph import LLMs, crear_grafo
# Importa el contrato de decisión y el estado inicial
from app.state import Decision, estado_inicial
# Importa los guardarraíles del supervisor
from app.supervisor import aplicar_reglas
# Importa las validaciones
from app.validation import validar_analisis, validar_investigacion
# Importa los dobles de prueba compartidos
from tests.fakes import (
    PREGUNTA, LLMGuionado, armar_orquestador, decidir, guion_analista, guion_investigador,
    guion_supervisor_feliz, guion_sintetizador, pedir, settings_de_prueba,
)

# ---------- Tests ----------


# Corre el orquestador y devuelve (estado final, lista de nodos en el orden en que se ejecutaron)
def correr(app: Any, pregunta: str = PREGUNTA) -> tuple[dict[str, Any], list[str]]:
    # Recorre el flujo y junta los nodos ejecutados y el estado final
    async def _correr() -> tuple[dict[str, Any], list[str]]:
        nodos: list[str] = []
        final: dict[str, Any] = {}
        async for modo, contenido in app.astream(estado_inicial(pregunta), config={"recursion_limit": 60}, stream_mode=["updates", "values"]):
            if modo == "updates":
                nodos.extend(contenido.keys())
            else:
                final = contenido
        return final, nodos

    return asyncio.run(_correr())


# Texto del primer mensaje humano que recibió un LLM falso en una de sus llamadas
def humano_de(llm: LLMGuionado, llamada: int) -> str:
    for mensaje in llm.recibidos[llamada]:
        if isinstance(mensaje, HumanMessage):
            return str(mensaje.content)
    return ""


# ---------- Flujo principal ----------


# Test 1: delegación completa (investigador → analista → síntesis) con validación entre cada paso
def test_delegacion_completa() -> None:
    app, _ = armar_orquestador(guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador())
    estado, nodos = correr(app)
    # El orden de ejecución es exactamente el de la consigna, con la validación después de cada especialista
    assert nodos == ["supervisor", "investigador", "validacion", "supervisor", "analista", "validacion", "supervisor", "sintetizador"]
    # Ambos resultados pasaron la validación
    assert estado["investigacion_valida"] and estado["analisis_valido"]
    # Los datos son los reales de la base y los números los calcularon las herramientas
    assert [h["doc_id"] for h in estado["investigacion"]["hallazgos"]] == ["d01", "d02"]
    assert estado["analisis"]["rating_promedio"] == 3.5
    assert estado["analisis"]["sentimiento_global"] == "neutro"
    # La respuesta final llegó al estado y al historial de mensajes
    assert estado["respuesta_final"]
    assert isinstance(estado["messages"][-1], AIMessage) and estado["messages"][-1].content == estado["respuesta_final"]


# Test 2: el estado rastrea qué agente contribuyó con qué (y la validación deja su veredicto)
def test_estado_rastrea_contribuciones() -> None:
    app, _ = armar_orquestador(guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador())
    estado, _ = correr(app)
    agentes = [c["agente"] for c in estado["contribuciones"]]
    assert agentes == ["investigador", "validacion", "analista", "validacion", "sintetizador"]
    # El aporte del investigador incluye las herramientas que usó (procedencia)
    herramientas = [p["herramienta"] for p in estado["contribuciones"][0]["pasos"] if p["tipo"] == "decide_herramienta"]
    assert herramientas == ["buscar_resenas"]
    # Cada decisión del supervisor queda auditada
    assert [d["final"] for d in estado["decisiones"]] == ["investigador", "analista", "sintetizador"]


# Test 3: contexto aislado: cada especialista ve SOLO su instrucción, no la pregunta original ni el historial
def test_contexto_aislado() -> None:
    app, llms = armar_orquestador(guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador())
    correr(app)
    # El investigador recibió: 1 mensaje de sistema (su rol) y 1 humano con la instrucción del supervisor
    primera = llms.investigador.recibidos[0]
    assert [type(m).__name__ for m in primera] == ["SystemMessage", "HumanMessage"]
    assert humano_de(llms.investigador, 0) == "Buscá reseñas sobre la entrega de los auriculares."
    # El analista recibió su instrucción + las reseñas, pero NO la pregunta original del usuario
    texto_analista = humano_de(llms.analista, 0)
    assert "Analizá el sentimiento" in texto_analista and "d01" in texto_analista
    assert PREGUNTA not in texto_analista
    # El supervisor recibe un resumen del estado, no el historial de mensajes de los especialistas
    resumen = humano_de(llms.supervisor, 1)
    assert "Investigación: VALIDADA" in resumen and "tool" not in resumen.lower()


# Test 3b: el sintetizador recibe datos estructurados (cantidad y hallazgos reales), no el resumen en texto libre del investigador
def test_sintesis_usa_datos_estructurados() -> None:
    app, llms = armar_orquestador(guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador())
    correr(app)
    datos = humano_de(llms.sintetizador, 0)
    assert '"cantidad_resenas": 2' in datos and '"hallazgos"' in datos and '"analisis"' in datos
    assert "Encontré reseñas de entrega" not in datos


# ---------- Supervisor infinito y guardarraíles ----------


# Test 4: un supervisor que nunca termina se corta por el tope de pasos (no llega al límite de recursión)
def test_supervisor_infinito_se_corta_por_tope_de_pasos() -> None:
    # Investigador que nunca encuentra nada + muchos refinamientos permitidos: solo el tope de pasos puede frenarlo
    configuracion = settings_de_prueba(max_pasos_supervisor=4, max_refinamientos=100)
    app, llms = armar_orquestador(
        [decidir("investigador")], [AIMessage(content="No encontré nada.")], [AIMessage(content="-")], [AIMessage(content="Sin datos suficientes.")],
        configuracion=configuracion, repetir_supervisor=True,
    )
    llms.investigador.repetir_ultima = True
    estado, nodos = correr(app)
    # El supervisor decidió exactamente 4 veces y la última fue el cierre forzado
    assert estado["pasos_supervisor"] == 4
    assert estado["decisiones"][-1]["final"] == "sintetizador"
    assert "tope" in estado["decisiones"][-1]["guardarrail"]
    # El investigador corrió 3 veces (pasos 1 a 3) y el flujo terminó con una respuesta
    assert estado["intentos_investigador"] == 3
    assert nodos[-1] == "sintetizador" and estado["respuesta_final"]


# Test 5: el supervisor pide analizar sin investigación → el guardarraíl lo redirige al investigador
def test_guardarrail_no_analizar_sin_investigacion() -> None:
    supervisor = [decidir("analista"), decidir("analista"), decidir("sintetizador")]
    app, _ = armar_orquestador(supervisor, guion_investigador(), guion_analista(), guion_sintetizador())
    estado, nodos = correr(app)
    assert estado["decisiones"][0]["propuesta"] == "analista" and estado["decisiones"][0]["final"] == "investigador"
    assert "sin una investigación validada" in estado["decisiones"][0]["guardarrail"]
    assert nodos[1] == "investigador"
    assert estado["analisis_valido"]


# Test 6: el supervisor intenta cerrar antes de tiempo → la rúbrica lo obliga a completar el trabajo
def test_guardarrail_no_cerrar_sin_rubrica() -> None:
    supervisor = [decidir("sintetizador"), decidir("analista"), decidir("sintetizador")]
    app, _ = armar_orquestador(supervisor, guion_investigador(), guion_analista(), guion_sintetizador())
    estado, _ = correr(app)
    assert estado["decisiones"][0]["final"] == "investigador"
    assert "rúbrica no se cumple" in estado["decisiones"][0]["guardarrail"]
    assert estado["investigacion_valida"] and estado["analisis_valido"]


# Test 7: si el LLM del supervisor falla, decide la regla por defecto y el flujo se completa igual
def test_supervisor_con_llm_caido_usa_regla_por_defecto() -> None:
    app, _ = armar_orquestador([RuntimeError("caído")], guion_investigador(), guion_analista(), guion_sintetizador(), repetir_supervisor=True)
    estado, nodos = correr(app)
    assert [d["propuesta"] for d in estado["decisiones"]] == ["(falló el LLM)"] * 3
    assert nodos.count("investigador") == 1 and nodos.count("analista") == 1 and nodos[-1] == "sintetizador"
    assert estado["investigacion_valida"] and estado["analisis_valido"]


# ---------- Validación, refinamiento y conflictos ----------


# Test 8: la investigación sin resultados es rechazada y el supervisor reintenta informando el error específico
def test_refinamiento_de_investigacion_con_error_especifico() -> None:
    investigador = [pedir(("buscar_resenas", {"consulta": "drones"}), prefijo="x"), AIMessage(content="No hay nada sobre drones.")] + guion_investigador()
    supervisor = [decidir("investigador", "Buscá reseñas de drones."), decidir("investigador", "Probá de nuevo."), decidir("analista"), decidir("sintetizador")]
    app, llms = armar_orquestador(supervisor, investigador, guion_analista(), guion_sintetizador())
    estado, _ = correr(app)
    # El primer intento fue rechazado con un motivo concreto
    assert estado["contribuciones"][1]["agente"] == "validacion" and not estado["contribuciones"][1]["valido"]
    # El segundo intento recibió ese motivo en su instrucción y se corrigió
    segunda_instruccion = humano_de(llms.investigador, 2)
    assert "Corregí este problema" in segunda_instruccion and "no se encontró ninguna reseña" in segunda_instruccion
    assert estado["intentos_investigador"] == 2 and estado["investigacion_valida"] and estado["analisis_valido"]


# Test 9: conflicto de datos: el analista informa un rating que no coincide con la investigación → el árbitro lo rechaza y se corrige
def test_conflicto_numerico_el_arbitro_rechaza() -> None:
    analista = guion_analista(rating_alterado=1.0) + guion_analista()
    supervisor = [decidir("investigador"), decidir("analista"), decidir("analista"), decidir("sintetizador")]
    app, llms = armar_orquestador(supervisor, guion_investigador(), analista, guion_sintetizador())
    estado, _ = correr(app)
    veredictos = [c for c in estado["contribuciones"] if c["agente"] == "validacion" and c["sobre"] == "analista"]
    assert not veredictos[0]["valido"] and "no coincide con el real (3.50)" in veredictos[0]["resumen"]
    assert veredictos[1]["valido"]
    assert estado["intentos_analista"] == 2 and estado["analisis"]["rating_promedio"] == 3.5
    # El reintento del analista recibió el error específico
    assert "no coincide con el real" in humano_de(llms.analista, 3)


# Test 10: refinamientos agotados → degradación honesta: se cierra sin analizar y la síntesis informa la limitación
def test_degradacion_honesta_al_agotar_intentos() -> None:
    sin_datos = [pedir(("buscar_resenas", {"consulta": "drones"}), prefijo="x"), AIMessage(content="Nada.")]
    supervisor = [decidir("investigador")] * 3
    app, llms = armar_orquestador(supervisor, sin_datos * 2, guion_analista(), ["Respuesta explicando la limitación."])
    estado, nodos = correr(app)
    assert estado["intentos_investigador"] == 2 and estado["intentos_analista"] == 0 and "analista" not in nodos
    assert not estado["investigacion_valida"]
    assert "agotaron los intentos" in estado["decisiones"][-1]["guardarrail"]
    # El sintetizador recibió la limitación y ningún dato sin validar
    datos = humano_de(llms.sintetizador, 0)
    assert "No se pudo validar la investigación" in datos and '"investigacion":' not in datos


# Test 11: un especialista que falla (excepción del LLM) no rompe el grafo: queda registrado y la validación lo rechaza
def test_falla_de_especialista_no_rompe_el_grafo() -> None:
    configuracion = settings_de_prueba(max_refinamientos=0)
    app, _ = armar_orquestador([decidir("investigador")] * 2, [RuntimeError("API caída")], guion_analista(), ["Cierre con limitación."], configuracion=configuracion)
    estado, nodos = correr(app)
    assert "RuntimeError" in estado["contribuciones"][0]["error"]
    assert estado["error_investigacion"] == "el investigador no entregó ningún resultado"
    assert nodos[-1] == "sintetizador"


# Test 12: un especialista atrapado en su propio bucle ReAct se corta por su límite de iteraciones
def test_limite_de_iteraciones_del_especialista() -> None:
    configuracion = settings_de_prueba(max_refinamientos=0, limite_iteraciones_agente=2)
    bucle = LLMGuionado(respuestas=[pedir(("buscar_resenas", {"consulta": "entrega"}))], repetir_ultima=True)
    app, llms = armar_orquestador([decidir("investigador")] * 2, [], guion_analista(), ["Cierre."], configuracion=configuracion)
    # Reemplaza el LLM del investigador por uno que pide herramientas para siempre
    llms2 = LLMs(supervisor=llms.supervisor, investigador=bucle, analista=llms.analista, sintetizador=llms.sintetizador)
    estado, _ = correr(crear_grafo(llms2, configuracion).compile())
    assert "límite de 2 iteraciones" in estado["contribuciones"][0]["error"]


# ---------- Reglas del supervisor (unitarias) ----------


# Test 13: tabla de decisiones del guardarraíl
@pytest.mark.parametrize(
    ("propuesta", "estado", "esperado"),
    [
        ("investigador", {}, "investigador"),
        ("analista", {}, "investigador"),
        ("sintetizador", {}, "investigador"),
        ("sintetizador", {"investigacion_valida": True, "intentos_investigador": 1}, "analista"),
        ("investigador", {"investigacion_valida": True, "intentos_investigador": 1}, "analista"),
        ("sintetizador", {"investigacion_valida": True, "analisis_valido": True, "intentos_investigador": 1, "intentos_analista": 1}, "sintetizador"),
        ("investigador", {"intentos_investigador": 2}, "sintetizador"),
        ("analista", {"investigacion_valida": True, "intentos_investigador": 1, "intentos_analista": 2}, "sintetizador"),
    ],
)
def test_tabla_de_guardarrailes(propuesta: str, estado: dict[str, Any], esperado: str) -> None:
    decision, _ = aplicar_reglas(Decision(siguiente=propuesta, instruccion="x", razon="r"), estado, max_intentos=2)  # type: ignore[arg-type]
    assert decision.siguiente == esperado


# ---------- Herramientas, base de conocimiento y validación (unitarias) ----------


# Test 14: la búsqueda encuentra lo relevante, ignora tildes y devuelve vacío si no hay nada
def test_busqueda_en_la_base() -> None:
    ids = [d.id for d in asyncio.run(knowledge_base.buscar("ENTREGA auriculares", 2))]
    assert ids == ["d01", "d02"]
    assert [d.id for d in asyncio.run(knowledge_base.buscar("entrega cafetera"))] == ["d10"]
    # Deben coincidir TODAS las palabras: una palabra que no existe anula la búsqueda en vez de devolver reseñas de otros temas
    assert asyncio.run(knowledge_base.buscar("drones")) == []
    assert asyncio.run(knowledge_base.buscar("drones calidad")) == []
    assert asyncio.run(knowledge_base.buscar("los clientes")) == []


# Test 15: las herramientas devuelven JSON útil; "sin resultados" es una respuesta válida (no un error) y avisa qué hacer
def test_herramientas_de_investigacion() -> None:
    sin_datos = json.loads(asyncio.run(buscar_resenas.ainvoke({"consulta": "drones"})))
    assert sin_datos["estado"] == "sin_resultados" and sin_datos["resultados"] == [] and "no hay datos" in sin_datos["mensaje"]
    assert '"doc_id": "d03"' in asyncio.run(buscar_resenas.ainvoke({"consulta": "sonido batería"}))


# Test 16: sentimiento (con negación) y estadísticas
def test_herramientas_de_analisis() -> None:
    assert puntuar_sentimiento("Es excelente y rápido") == 1.0
    assert puntuar_sentimiento("Es pésimo y llegó tarde") == -1.0
    assert puntuar_sentimiento("No es bueno") == -1.0
    assert puntuar_sentimiento("Llegó el martes") == 0.0
    assert etiquetar(0.0) == "neutro" and etiquetar(0.5) == "positivo" and etiquetar(-0.5) == "negativo"
    assert '"promedio": 3.5' in asyncio.run(calcular_estadisticas.ainvoke({"nombre": "rating", "valores": [5, 2]}))


# Test 17: la validación rechaza hallazgos inventados y análisis que no cubren la investigación
def test_validacion_de_procedencia_y_cobertura() -> None:
    inventado = {"consulta": "x" * 5, "resumen": "r", "hallazgos": [{"doc_id": "d01", "producto": "p", "categoria": "c", "rating": 5, "texto": "texto inventado"}]}
    assert "no coincide con ningún documento real" in validar_investigacion(inventado)
    d1 = knowledge_base.obtener("d01")
    assert d1 is not None
    real = {"consulta": "entrega", "resumen": "r", "hallazgos": [{"doc_id": "d01", "producto": d1.producto, "categoria": d1.categoria, "rating": d1.rating, "texto": d1.texto}]}
    assert validar_investigacion(real) == ""
    incompleto = {"sentimientos": [{"doc_id": "d02", "puntaje": -1.0, "etiqueta": "negativo"}], "puntaje_promedio": -1.0, "rating_promedio": 5.0, "cantidad_analizada": 1, "sentimiento_global": "negativo", "conclusiones": "c"}
    assert "no cubre las mismas reseñas" in validar_analisis(incompleto, real)
    assert "faltan o son inválidos" in validar_analisis({"sentimientos": [], "conclusiones": ""}, real)


# ---------- Configuración y diagrama ----------


# Test 18: la configuración valida las variables al arrancar
def test_configuracion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for variable in ("LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_MODEL", "MAX_REFINAMIENTOS", "MAX_PASOS_SUPERVISOR"):
        monkeypatch.delenv(variable, raising=False)
    # Sin API key corta con un mensaje claro
    with pytest.raises(config.ConfigError, match="OPENAI_API_KEY"):
        config.cargar_configuracion()
    # El valor de ejemplo del .env.example tampoco sirve
    monkeypatch.setenv("OPENAI_API_KEY", "tu_api_key_aqui")
    with pytest.raises(config.ConfigError):
        config.cargar_configuracion()
    # Con key real usa los valores por defecto y no muestra la key en el repr
    monkeypatch.setenv("OPENAI_API_KEY", "sk-prueba")
    ajustes = config.cargar_configuracion()
    assert ajustes.max_refinamientos == 1 and ajustes.max_pasos_supervisor == 8 and ajustes.modelo == "gpt-4o-mini"
    assert "sk-prueba" not in repr(ajustes)
    # MAX_REFINAMIENTOS=0 es válido (sin reintentos) y un valor inválido se rechaza
    monkeypatch.setenv("MAX_REFINAMIENTOS", "0")
    assert config.cargar_configuracion().max_refinamientos == 0
    monkeypatch.setenv("MAX_PASOS_SUPERVISOR", "1")
    with pytest.raises(config.ConfigError, match="MAX_PASOS_SUPERVISOR"):
        config.cargar_configuracion()
    # Un RECURSION_LIMIT demasiado bajo para el tope de pasos del supervisor se rechaza al arrancar (por ejemplo, copiado de otra pre-entrega)
    monkeypatch.setenv("MAX_PASOS_SUPERVISOR", "8")
    monkeypatch.setenv("RECURSION_LIMIT", "10")
    with pytest.raises(config.ConfigError, match="RECURSION_LIMIT"):
        config.cargar_configuracion()


# Test 19: el diagrama Mermaid del grafo contiene los nodos y las aristas condicionales del supervisor
def test_diagrama_mermaid() -> None:
    app, _ = armar_orquestador([], [], [], [])
    mermaid = app.get_graph().draw_mermaid()
    for nodo in ("supervisor", "investigador", "analista", "validacion", "sintetizador", "aprobacion", "despacho"):
        assert nodo in mermaid
    assert "supervisor -.-> investigador" in mermaid and "validacion --> supervisor" in mermaid
    # Tramo HITL: solo las tareas críticas pasan por la aprobación humana
    assert "sintetizador -.-> aprobacion" in mermaid and "aprobacion --> despacho" in mermaid
