# Tests del flujo Human-in-the-loop: la tarea crítica se PAUSA antes del efecto secundario y solo sigue con una decisión humana
# Se usan LLM falsos (guiones) y un checkpointer en memoria; el grafo, el interrupt y la reanudación son los reales

# Importa Any para tipar
from typing import Any

# Importa el checkpointer en memoria (en producción es Redis; la lógica de pausa y reanudación es la misma)
from langgraph.checkpoint.memory import InMemorySaver
# Importa Command para reanudar el grafo pausado
from langgraph.types import Command

# Importa la regla de criticidad y la arista condicional
from app.hitl import motivo_criticidad, ruta_tras_sintesis
# Importa el estado inicial del orquestador
from app.state import estado_inicial
# Importa los dobles de prueba
from tests.fakes import (
    PREGUNTA, DespachadorEspia, armar_orquestador, guion_analista, guion_investigador, guion_supervisor_feliz, guion_sintetizador,
)

# Destinatario de los tests
DESTINO = "gerencia@empresa.com"


# Arma un orquestador con checkpointer y despachador espía
def armar() -> tuple[Any, DespachadorEspia, InMemorySaver]:
    espia = DespachadorEspia()
    memoria = InMemorySaver()
    grafo, _ = armar_orquestador(
        guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador(), checkpointer=memoria, despachador=espia
    )
    return grafo, espia, memoria


# Config de LangGraph de un hilo (en la API, el thread_id es el job_id)
def config_de(hilo: str) -> Any:
    return {"configurable": {"thread_id": hilo}, "recursion_limit": 60}


# Corre el grafo hasta que termina o se pausa (descarta los eventos)
async def correr(grafo: Any, entrada: Any, hilo: str) -> None:
    async for _ in grafo.astream(entrada, config=config_de(hilo), stream_mode="updates"):
        pass


# Test 1: una tarea crítica se pausa en la aprobación y NO envía nada antes de que alguien decida
async def test_tarea_critica_se_pausa_antes_del_efecto_secundario() -> None:
    grafo, espia, _ = armar()
    await correr(grafo, estado_inicial(PREGUNTA, DESTINO), "t1")
    foto = await grafo.aget_state(config_de("t1"))
    assert foto.next == ("aprobacion",)
    assert espia.enviados == []
    # El revisor recibe todo lo que necesita para decidir: qué se va a enviar, a quién y por qué es crítico
    pendiente = foto.tasks[0].interrupts[0].value
    assert pendiente["destinatario"] == DESTINO and pendiente["borrador"] == foto.values["respuesta_final"]
    assert "efecto secundario" in pendiente["motivo"]


# Test 2: con la aprobación el grafo se reanuda y envía exactamente una vez el informe aprobado
async def test_aprobacion_reanuda_y_envia() -> None:
    grafo, espia, _ = armar()
    await correr(grafo, estado_inicial(PREGUNTA, DESTINO), "t2")
    await correr(grafo, Command(resume={"aprobado": True, "comentario": "ok, mandalo"}), "t2")
    foto = await grafo.aget_state(config_de("t2"))
    assert foto.next == ()
    assert len(espia.enviados) == 1
    destinatario, _asunto, cuerpo = espia.enviados[0]
    assert destinatario == DESTINO and cuerpo == foto.values["respuesta_final"]
    assert foto.values["aprobacion"] == {"estado": "aprobada", "comentario": "ok, mandalo"}
    assert "informe enviado a gerencia@empresa.com" in foto.values["despacho"]


# Test 3: con el rechazo el grafo termina sin enviar nada
async def test_rechazo_no_envia() -> None:
    grafo, espia, _ = armar()
    await correr(grafo, estado_inicial(PREGUNTA, DESTINO), "t3")
    await correr(grafo, Command(resume={"aprobado": False, "comentario": "falta revisar los números"}), "t3")
    foto = await grafo.aget_state(config_de("t3"))
    assert foto.next == () and espia.enviados == []
    assert foto.values["aprobacion"]["estado"] == "rechazada" and "cancelado" in foto.values["despacho"]


# Test 4: una tarea normal (sin envío) nunca se pausa ni pasa por la aprobación
async def test_tarea_normal_no_se_pausa() -> None:
    grafo, espia, _ = armar()
    nodos: list[str] = []
    async for evento in grafo.astream(estado_inicial(PREGUNTA), config=config_de("t4"), stream_mode="updates"):
        nodos.extend(evento.keys())
    foto = await grafo.aget_state(config_de("t4"))
    assert foto.next == () and "aprobacion" not in nodos and "despacho" not in nodos and espia.enviados == []


# Test 5: la pausa sobrevive a un "reinicio": otro grafo (otro proceso) con el mismo checkpointer retoma el trabajo donde quedó
async def test_la_reanudacion_funciona_desde_otro_grafo() -> None:
    grafo_a, _, memoria = armar()
    await correr(grafo_a, estado_inicial(PREGUNTA, DESTINO), "t5")
    # Un grafo nuevo, con LLM nuevos (sin guion: no debería necesitarlos) y el MISMO almacén de checkpoints
    espia_b = DespachadorEspia()
    grafo_b, _ = armar_orquestador([], [], [], [], checkpointer=memoria, despachador=espia_b)
    await correr(grafo_b, Command(resume={"aprobado": True}), "t5")
    assert len(espia_b.enviados) == 1


# Test 6: la regla de criticidad es explícita y se puede probar sola
def test_regla_de_criticidad() -> None:
    assert motivo_criticidad(estado_inicial(PREGUNTA)) == ""
    assert motivo_criticidad(estado_inicial(PREGUNTA, "   ")) == ""
    assert DESTINO in motivo_criticidad(estado_inicial(PREGUNTA, DESTINO))
    assert ruta_tras_sintesis(estado_inicial(PREGUNTA)) == "fin"
    assert ruta_tras_sintesis(estado_inicial(PREGUNTA, DESTINO)) == "aprobacion"
