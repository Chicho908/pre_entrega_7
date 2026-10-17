# Tests de las correcciones del dominio (orquestador de la PE6) que salieron de leer las trazas en Phoenix:
#   1. la validación aceptaba solo el promedio global y rechazaba comparaciones que informaban el de un producto
#   2. "la fuente se cayó" y "no hay datos" eran el mismo error; ahora son distintos y el reintento sabe cuál es

# Importa json para armar los resultados de herramientas
import json

# Importa pytest para el monkeypatch
import pytest
# Importa los mensajes de herramientas
from langchain_core.messages import AIMessage, ToolMessage

# Importa las funciones a probar
from app import knowledge_base
# Importa la construcción del análisis a probar
from app.agents.analyst_agent import construir_analisis
# Importa el detector de errores de credenciales
from app.llm import es_error_de_credenciales
# Importa la validación del análisis
from app.validation import validar_analisis
# Importa los dobles de prueba
from tests.fakes import PREGUNTA, armar_orquestador, decidir, guion_sintetizador, pedir, settings_de_prueba
# Importa el estado inicial del orquestador
from app.state import estado_inicial

# Reseñas de dos productos: auriculares (d01=5, d02=2 → promedio 3.5) y zapatillas (d06=5, d07=4 → promedio 4.5); global 4.0
HALLAZGOS = [
    {"doc_id": doc_id, "producto": producto, "categoria": categoria, "rating": rating, "texto": texto}
    for doc_id, producto, categoria, rating, texto in [
        ("d01", "auriculares Pulse", "entrega", 5, knowledge_base.obtener("d01").texto),  # type: ignore[union-attr]
        ("d02", "auriculares Pulse", "entrega", 2, knowledge_base.obtener("d02").texto),  # type: ignore[union-attr]
        ("d06", "zapatillas Runner", "calidad", 5, knowledge_base.obtener("d06").texto),  # type: ignore[union-attr]
        ("d07", "zapatillas Runner", "entrega", 4, knowledge_base.obtener("d07").texto),  # type: ignore[union-attr]
    ]
]
INVESTIGACION = {"consulta": "comparar auriculares y zapatillas", "hallazgos": HALLAZGOS, "resumen": "4 reseñas"}


# Arma un análisis con los promedios que se le indiquen (sentimientos: d01 +1, d02 -1, d06 +1, d07 +0.5 → global 0.38, auriculares 0, zapatillas 0.75)
def analisis_con(rating: float, puntaje: float) -> dict[str, object]:
    return {
        "sentimientos": [
            {"doc_id": "d01", "puntaje": 1.0, "etiqueta": "positivo"}, {"doc_id": "d02", "puntaje": -1.0, "etiqueta": "negativo"},
            {"doc_id": "d06", "puntaje": 1.0, "etiqueta": "positivo"}, {"doc_id": "d07", "puntaje": 0.5, "etiqueta": "positivo"},
        ],
        "puntaje_promedio": puntaje, "rating_promedio": rating, "cantidad_analizada": 4, "sentimiento_global": "neutro", "conclusiones": "c",
    }


# Test 1: en una comparación se acepta el promedio global O el de cualquier producto (antes se rechazaba el de un producto)
def test_validacion_acepta_promedio_global_o_por_producto() -> None:
    assert validar_analisis(analisis_con(rating=4.0, puntaje=0.38), INVESTIGACION) == ""      # global
    assert validar_analisis(analisis_con(rating=4.5, puntaje=0.75), INVESTIGACION) == ""      # zapatillas
    assert validar_analisis(analisis_con(rating=3.5, puntaje=0.0), INVESTIGACION) == ""       # auriculares


# Test 2: un número que no corresponde a NINGÚN grupo sigue siendo rechazado
def test_validacion_sigue_rechazando_numeros_inventados() -> None:
    assert "rating_promedio" in validar_analisis(analisis_con(rating=2.0, puntaje=0.38), INVESTIGACION)
    assert "puntaje_promedio" in validar_analisis(analisis_con(rating=4.0, puntaje=0.9), INVESTIGACION)


# Arma el resultado de una herramienta calcular_estadisticas
def estadistica(id_llamada: str, nombre: str, cantidad: int, promedio: float) -> ToolMessage:
    cuerpo = {"nombre": nombre, "cantidad": cantidad, "promedio": promedio, "minimo": 1, "maximo": 5}
    return ToolMessage(content=json.dumps(cuerpo), name="calcular_estadisticas", tool_call_id=id_llamada)


# Test 3: si el analista calculó promedios por producto Y el global, se informa el global (el que cubre todas las reseñas)
def test_el_analisis_informa_la_estadistica_global() -> None:
    mensajes = [
        estadistica("a", "rating", 2, 3.5), estadistica("b", "rating", 2, 4.5),   # por producto
        estadistica("c", "rating", 4, 4.0),                                       # global
        estadistica("d", "sentimiento", 4, 0.38),
        ToolMessage(content=json.dumps({"doc_id": "d01", "puntaje": 1.0, "etiqueta": "positivo"}), name="analizar_sentimiento", tool_call_id="e"),
        ToolMessage(content=json.dumps({"doc_id": "d02", "puntaje": -1.0, "etiqueta": "negativo"}), name="analizar_sentimiento", tool_call_id="f"),
        ToolMessage(content=json.dumps({"doc_id": "d06", "puntaje": 1.0, "etiqueta": "positivo"}), name="analizar_sentimiento", tool_call_id="g"),
        ToolMessage(content=json.dumps({"doc_id": "d07", "puntaje": 0.5, "etiqueta": "positivo"}), name="analizar_sentimiento", tool_call_id="h"),
        AIMessage(content="Conclusiones."),
    ]
    analisis = construir_analisis(mensajes)
    assert analisis["rating_promedio"] == 4.0 and analisis["puntaje_promedio"] == 0.38 and analisis["cantidad_analizada"] == 4


# Corre un orquestador cuyo investigador busca "entrega auriculares" una sola vez y devuelve el estado final
async def investigar_una_vez() -> dict[str, object]:
    configuracion = settings_de_prueba(max_refinamientos=0)
    investigador = [pedir(("buscar_resenas", {"consulta": "entrega auriculares"}), prefijo="i"), AIMessage(content="No pude traer datos.")]
    grafo, _ = armar_orquestador([decidir("investigador"), decidir("sintetizador")], investigador, [], guion_sintetizador(), configuracion=configuracion)
    final: dict[str, object] = await grafo.ainvoke(estado_inicial(PREGUNTA), config={"recursion_limit": 60})
    return final


# Test 4: si la fuente se cae, la validación lo dice como FALLA TÉCNICA (no como "no hay datos")
async def test_fuente_caida_se_distingue_de_sin_resultados(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fuente de datos que falla como si la base estuviera caída
    async def buscar_roto(consulta: str, max_resultados: int = 5) -> list[knowledge_base.Documento]:
        raise ConnectionError("la base no responde")

    monkeypatch.setattr(knowledge_base, "buscar", buscar_roto)
    estado = await investigar_una_vez()
    assert estado["investigacion_valida"] is False
    assert "no respondió" in str(estado["error_investigacion"]) and "falla técnica" in str(estado["error_investigacion"])
    assert estado["investigacion"]["fuente_caida"] is True  # type: ignore[index]


# Test 5: cuando la fuente funciona pero no hay reseñas, el mensaje es "no hay datos" y no se marca como caída
async def test_sin_resultados_no_es_fuente_caida() -> None:
    configuracion = settings_de_prueba(max_refinamientos=0)
    investigador = [pedir(("buscar_resenas", {"consulta": "drones"}), prefijo="i"), AIMessage(content="No hay datos.")]
    grafo, _ = armar_orquestador([decidir("investigador"), decidir("sintetizador")], investigador, [], guion_sintetizador(), configuracion=configuracion)
    estado = await grafo.ainvoke(estado_inicial("¿Qué opinan de los drones?"), config={"recursion_limit": 60})
    assert "no se encontró ninguna reseña" in estado["error_investigacion"] and estado["investigacion"]["fuente_caida"] is False


# Test 6: un error 401/403 (también envuelto en otro) se reconoce como falla de credenciales; el resto no
def test_detecta_errores_de_credenciales() -> None:
    # Error de prueba con status_code 401
    class Error401(Exception):
        status_code = 401

    envuelto = RuntimeError("falló")
    envuelto.__cause__ = Error401()
    assert es_error_de_credenciales(Error401()) and es_error_de_credenciales(envuelto)
    assert not es_error_de_credenciales(RuntimeError("red caída")) and not es_error_de_credenciales(TimeoutError())
