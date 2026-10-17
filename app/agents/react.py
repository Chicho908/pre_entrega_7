# Módulo base de los especialistas: construye un agente ReAct (modelo <-> herramientas) y lo ejecuta de forma segura
# Cada especialista es un SUBGRAFO con su propio estado privado: sus mensajes internos NO se mezclan con los del grafo principal

# Importa Sequence para tipar la lista de herramientas
from collections.abc import Awaitable, Sequence
# Importa Any y Protocol para tipar los nodos
from typing import Any, Protocol

# Importa BaseChatModel para tipar el LLM
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los mensajes que usa el agente
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
# Importa BaseTool para tipar las herramientas
from langchain_core.tools import BaseTool, ToolException
# Importa los bloques para armar el subgrafo; MessagesState ya trae messages con el reducer add_messages
from langgraph.graph import END, START, MessagesState, StateGraph
# Importa el error que LangGraph lanza al superar el límite de recursión
from langgraph.errors import GraphRecursionError
# Importa el tipo del grafo compilado
from langgraph.graph.state import CompiledStateGraph
# Importa el detector de errores de credenciales del LLM
from app.llm import es_error_de_credenciales
# Importa el estado compartido (los nodos del grafo principal lo reciben como entrada)
from app.state import EstadoOrquestador
# Importa ToolNode (ejecuta las herramientas pedidas) y tools_condition (decide si ir a herramientas o terminar)
from langgraph.prebuilt import ToolNode, tools_condition


# Tipo de un nodo asíncrono del grafo principal: recibe el estado compartido y devuelve solo los campos que actualiza
class Nodo(Protocol):
    # Los nodos se llaman con el estado (el parámetro debe llamarse state para cumplir el protocolo de LangGraph)
    def __call__(self, state: EstadoOrquestador) -> Awaitable[dict[str, Any]]: ...


# Error propio: el especialista no pudo terminar su trabajo (límite de pasos o falla del LLM)
class ErrorAgente(Exception):
    pass


# Tipo del agente ReAct compilado (su estado es solo la lista de mensajes)
AgenteReact = CompiledStateGraph[MessagesState, None, MessagesState, MessagesState]


# Convierte la excepción de una herramienta en el texto que lee el LLM (las fallas deliberadas de la herramienta pasan tal cual)
def mensaje_de_error_herramienta(error: Exception) -> str:
    # Una ToolException ya trae un mensaje pensado para el LLM (por ejemplo FUENTE_CAIDA)
    if isinstance(error, ToolException):
        return str(error)
    # Cualquier otra excepción (argumentos inválidos, bug) se informa con su tipo y detalle
    return f"Error: {error!r}\nCorregí los argumentos de la llamada."


# Construye y compila el agente ReAct de un especialista con sus herramientas acotadas
def crear_agente_react(
    llm: BaseChatModel,
    herramientas: Sequence[BaseTool],
    prompt_sistema: str,
) -> AgenteReact:
    # Vincula SOLO las herramientas de este especialista (mínimo privilegio)
    llm_con_herramientas = llm.bind_tools(herramientas)

    # Nodo del modelo: razona y decide si responde o pide una herramienta
    async def nodo_modelo(state: MessagesState) -> dict[str, list[BaseMessage]]:
        # Invoca al LLM con las instrucciones del rol (no se guardan en el estado) y el historial privado
        respuesta = await llm_con_herramientas.ainvoke([SystemMessage(content=prompt_sistema), *state["messages"]])
        # Devuelve solo la actualización: el reducer add_messages la agrega
        return {"messages": [respuesta]}

    # Crea el subgrafo con su propio estado de mensajes
    subgrafo = StateGraph(MessagesState)
    # Nodo que razona
    subgrafo.add_node("modelo", nodo_modelo)
    # Nodo que ejecuta herramientas; el handler convierte las excepciones en mensajes que el LLM puede leer (el span de la herramienta igual queda en ERROR)
    subgrafo.add_node("herramientas", ToolNode(herramientas, handle_tool_errors=mensaje_de_error_herramienta))
    # Arranca en el modelo
    subgrafo.add_edge(START, "modelo")
    # Si el modelo pidió herramientas va a "herramientas"; si no, termina
    subgrafo.add_conditional_edges("modelo", tools_condition, {"tools": "herramientas", END: END})
    # Cierra el ciclo ReAct: después de la herramienta vuelve al modelo
    subgrafo.add_edge("herramientas", "modelo")
    # Devuelve el subgrafo compilado
    return subgrafo.compile()


# Ejecuta un especialista con SOLO la instrucción recibida (sin historial ni metadata del sistema) y devuelve sus mensajes
async def ejecutar_agente(
    agente: AgenteReact,
    instruccion: str,
    limite_iteraciones: int,
) -> list[BaseMessage]:
    # Cada iteración ReAct usa 2 pasos del grafo (modelo + herramientas); +1 por el último paso del modelo
    limite_recursion = 2 * limite_iteraciones + 1
    # Intenta correr el ciclo con tope de pasos
    try:
        resultado = await agente.ainvoke({"messages": [HumanMessage(content=instruccion)]}, config={"recursion_limit": limite_recursion})
    # Si supera el tope de pasos, lo informa con un error propio en vez de colgarse
    except GraphRecursionError as error:
        raise ErrorAgente(f"superó el límite de {limite_iteraciones} iteraciones sin terminar") from error
    # Cualquier otra falla (red, API, etc.) se convierte en un error propio con su causa
    except Exception as error:
        # API key inválida: no es un problema del especialista sino de la configuración, así que el trabajo debe terminar en FAILED
        if es_error_de_credenciales(error):
            raise
        raise ErrorAgente(f"falló con {type(error).__name__}: {error}") from error
    # Devuelve los mensajes del ciclo (el primero es la instrucción)
    return list(resultado["messages"])
