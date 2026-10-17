# Dobles de prueba compartidos: un LLM falso que sigue un guion, guiones de ejemplo y el armado del orquestador
# Las herramientas, la validación, los guardarraíles y el grafo son los REALES; solo cambia el "cerebro" (el LLM)

# Importa asyncio para simular la demora del grafo falso
import asyncio
# Importa Sequence para tipar los mensajes que recibe el LLM falso
from collections.abc import Sequence
# Importa dataclass para definir los dobles simples
from dataclasses import dataclass
# Importa Any para tipar
from typing import Any

# Importa BaseChatModel para crear el LLM falso
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los tipos de mensajes
from langchain_core.messages import AIMessage, BaseMessage
# Importa los tipos de resultado de un modelo de chat
from langchain_core.outputs import ChatGeneration, ChatResult
# Importa el checkpointer en memoria y el tipo base
from langgraph.checkpoint.base import BaseCheckpointSaver
# Importa Field para declarar campos con valor inicial en el LLM falso
from pydantic import Field

# Importa la base de reseñas para armar los guiones con datos reales
from app import knowledge_base
# Importa las herramientas y funciones del analista
from app.agents.analyst_agent import puntuar_sentimiento
# Importa Settings para armar la configuración de prueba
from app.config import Settings
# Importa el grupo de LLM y el armado del grafo
from app.graph import LLMs, crear_grafo
# Importa el despachador
from app.hitl import Despachador
# Importa el contrato de decisión
from app.state import Decision

# ---------- LLM falso ----------


# Objeto que imita a llm.with_structured_output(Decision): devuelve decisiones del guion
class DecisorGuionado:
    # Guarda el guion y la lista donde registra lo que recibió
    def __init__(self, llm: "LLMGuionado") -> None:
        self.llm = llm

    # Devuelve la próxima decisión del guion (o lanza el error si el guion pone una excepción)
    async def ainvoke(self, mensajes: Sequence[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        # Registra lo que recibió para poder comprobarlo en los tests
        self.llm.recibidos.append(list(mensajes))
        # Elige la respuesta (la última si debe repetir)
        posicion = min(self.llm.indice, len(self.llm.respuestas) - 1) if self.llm.repetir_ultima else self.llm.indice
        self.llm.indice += 1
        respuesta = self.llm.respuestas[posicion]
        # Si el guion pone una excepción, la lanza (simula una falla del LLM)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


# LLM falso: sirve para los tres usos del proyecto (chat con herramientas, salida estructurada y texto simple)
class LLMGuionado(BaseChatModel):
    # Respuestas que va devolviendo, una por llamada (AIMessage, Decision o una Exception)
    respuestas: list[Any]
    # Si es True, repite la última respuesta indefinidamente (simula un supervisor que nunca termina)
    repetir_ultima: bool = False
    # Posición de la próxima respuesta
    indice: int = 0
    # Mensajes que recibió en cada llamada
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)

    # Identifica el tipo de modelo (requisito de BaseChatModel)
    @property
    def _llm_type(self) -> str:
        return "guionado"

    # bind_tools devuelve el mismo objeto: el guion ya decide cuándo "pedir" herramientas
    def bind_tools(self, tools: Any, **kwargs: Any) -> "LLMGuionado":
        return self

    # with_structured_output devuelve el decisor del guion
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return DecisorGuionado(self)

    # Devuelve la siguiente respuesta del guion
    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.recibidos.append(list(messages))
        posicion = min(self.indice, len(self.respuestas) - 1) if self.repetir_ultima else self.indice
        self.indice += 1
        respuesta = self.respuestas[posicion]
        if isinstance(respuesta, Exception):
            raise respuesta
        # Si el guion trae un texto suelto, lo convierte en un mensaje del modelo
        if isinstance(respuesta, str):
            respuesta = AIMessage(content=respuesta)
        # Devuelve una COPIA sin id: el reducer add_messages reemplaza mensajes con el mismo id
        copia = respuesta.model_copy(deep=True)
        copia.id = None
        return ChatResult(generations=[ChatGeneration(message=copia)])


# Arma un mensaje del modelo que pide una o varias herramientas a la vez
def pedir(*llamadas: tuple[str, dict[str, Any]], prefijo: str = "c") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": nombre, "args": args, "id": f"{prefijo}{i}", "type": "tool_call"} for i, (nombre, args) in enumerate(llamadas)],
    )


# Atajo para armar una decisión del supervisor
def decidir(destino: str, instruccion: str = "", razon: str = "decisión del guion") -> Decision:
    return Decision(siguiente=destino, instruccion=instruccion, razon=razon)  # type: ignore[arg-type]


# ---------- Guiones ----------


# Pregunta de ejemplo de los tests
PREGUNTA = "¿Qué opinan los clientes sobre la entrega de los auriculares y cómo es el sentimiento general?"
# Documentos que encuentra la búsqueda "entrega auriculares" con max_resultados=2
IDS_ENTREGA = ("d01", "d02")


# Configuración de prueba (no usa API key real)
def settings_de_prueba(**cambios: Any) -> Settings:
    base: dict[str, Any] = {
        "proveedor": "openai", "api_key": "no-se-usa", "modelo": "falso", "limite_recursion": 40,
        "max_pasos_supervisor": 8, "max_refinamientos": 1, "limite_iteraciones_agente": 8, "job_timeout_s": 30,
    }
    base.update(cambios)
    return Settings(**base)


# Guion del investigador: busca y resume
def guion_investigador(consulta: str = "entrega auriculares", max_resultados: int = 2) -> list[Any]:
    return [
        pedir(("buscar_resenas", {"consulta": consulta, "max_resultados": max_resultados}), prefijo="i"),
        AIMessage(content="Encontré reseñas de entrega de los auriculares: una muy positiva y una muy negativa."),
    ]


# Guion del analista: sentimiento por reseña + estadísticas (con valores correctos, o con un rating alterado para simular un error)
def guion_analista(ids: tuple[str, ...] = IDS_ENTREGA, rating_alterado: float | None = None) -> list[Any]:
    docs = [d for d in (knowledge_base.obtener(i) for i in ids) if d is not None]
    puntajes = [puntuar_sentimiento(d.texto) for d in docs]
    ratings = [float(d.rating) for d in docs]
    llamadas_sentimiento = [("analizar_sentimiento", {"doc_id": d.id, "texto": d.texto}) for d in docs]
    llamadas_stats = [
        ("calcular_estadisticas", {"nombre": "sentimiento", "valores": puntajes}),
        ("calcular_estadisticas", {"nombre": "rating", "valores": [rating_alterado] if rating_alterado is not None else ratings}),
    ]
    return [
        pedir(*llamadas_sentimiento, prefijo="s"),
        pedir(*llamadas_stats, prefijo="e"),
        AIMessage(content="El sentimiento es mixto: hay una experiencia excelente y otra pésima, con un rating promedio intermedio."),
    ]


# Guion del sintetizador: una respuesta de texto
def guion_sintetizador() -> list[Any]:
    return [AIMessage(content="Las opiniones sobre la entrega están divididas: 2 reseñas, sentimiento neutro y rating promedio de 3.5.")]


# Guion del supervisor del camino feliz: investigar → analizar → sintetizar
def guion_supervisor_feliz() -> list[Any]:
    return [
        decidir("investigador", "Buscá reseñas sobre la entrega de los auriculares.", "primero hay que conseguir los datos"),
        decidir("analista", "Analizá el sentimiento y calculá los promedios.", "ya hay investigación validada; falta analizar"),
        decidir("sintetizador", "", "rúbrica cumplida: investigación y análisis validados"),
    ]


# Los 4 LLM falsos de un orquestador (tipados como LLMGuionado para poder inspeccionar lo que recibieron)
@dataclass(frozen=True)
class LLMsGuionados:
    supervisor: LLMGuionado
    investigador: LLMGuionado
    analista: LLMGuionado
    sintetizador: LLMGuionado


# Arma el grafo compilado y los 4 LLM falsos a partir de los guiones
def armar_orquestador(
    supervisor: list[Any], investigador: list[Any], analista: list[Any], sintetizador: list[Any],
    configuracion: Settings | None = None, repetir_supervisor: bool = False,
    checkpointer: BaseCheckpointSaver[str] | None = None, despachador: Despachador | None = None,
) -> tuple[Any, LLMsGuionados]:
    # Un LLM falso por rol
    llms = LLMsGuionados(
        supervisor=LLMGuionado(respuestas=supervisor, repetir_ultima=repetir_supervisor),
        investigador=LLMGuionado(respuestas=investigador),
        analista=LLMGuionado(respuestas=analista),
        sintetizador=LLMGuionado(respuestas=sintetizador),
    )
    # Compila el grafo con la configuración de prueba
    grupo = LLMs(supervisor=llms.supervisor, investigador=llms.investigador, analista=llms.analista, sintetizador=llms.sintetizador)
    return crear_grafo(grupo, configuracion or settings_de_prueba(), despachador).compile(checkpointer=checkpointer), llms


# Despachador espía: en vez de enviar nada, registra a quién se le habría enviado qué
class DespachadorEspia:
    # Crea la lista donde se acumulan los envíos registrados
    def __init__(self) -> None:
        self.enviados: list[tuple[str, str, str]] = []

    # Registra el envío en vez de hacerlo y devuelve un id correlativo
    async def enviar(self, destinatario: str, asunto: str, cuerpo: str) -> str:
        self.enviados.append((destinatario, asunto, cuerpo))
        return f"espia{len(self.enviados)}"



# Foto mínima del estado del grafo que necesita el worker (lo que devuelve aget_state)
class FotoFalsa:
    # Guarda los valores del estado; sin nodos pendientes (el grafo terminó)
    def __init__(self, valores: dict[str, Any]) -> None:
        self.next: tuple[str, ...] = ()
        self.tasks: tuple[Any, ...] = ()
        self.values = valores


# Grafo falso para probar el worker sin LLM: puede tardar, fallar o responder algo fijo
class GrafoFalso:
    # Configura cuánto tarda, si falla y qué responde; lleva la cuenta de ejecuciones y de las que corren a la vez
    def __init__(self, demora_s: float = 0.0, error: Exception | None = None, respuesta: str = "respuesta falsa") -> None:
        self.demora_s = demora_s
        self.error = error
        self.respuesta = respuesta
        self.ejecuciones = 0
        self.en_paralelo = 0
        self.maximo_en_paralelo = 0

    # Simula la ejecución del grafo: cuenta las ejecuciones en paralelo, espera, falla o termina sin emitir eventos
    async def astream(self, *args: Any, **kwargs: Any) -> Any:
        self.ejecuciones += 1
        self.en_paralelo += 1
        self.maximo_en_paralelo = max(self.maximo_en_paralelo, self.en_paralelo)
        try:
            await asyncio.sleep(self.demora_s)
            if self.error is not None:
                raise self.error
        finally:
            self.en_paralelo -= 1
        return
        yield  # convierte la función en un generador asíncrono (no emite eventos)

    # Devuelve la foto final del estado (sin pasos pendientes)
    async def aget_state(self, *args: Any, **kwargs: Any) -> FotoFalsa:
        return FotoFalsa({"respuesta_final": self.respuesta, "despacho": ""})
