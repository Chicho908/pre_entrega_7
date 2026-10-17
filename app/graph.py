# Módulo del grafo principal: une al supervisor, los especialistas, la validación, el sintetizador y la aprobación humana en un StateGraph
# Es el orquestador de la Pre-entrega 6 + un tramo final con Human-in-the-loop, y se compila con un checkpointer de Redis
#
# Topología jerárquica (el supervisor es el único que decide el flujo):
#   START → supervisor ─┬→ investigador ─→ validacion ─┐
#                       ├→ analista ─────→ validacion ─┤ (la validación siempre devuelve el control al supervisor)
#                       └→ sintetizador ─┬→ END (tarea normal)
#                                        └→ aprobacion ⏸ → despacho → END (tarea crítica: espera a un humano)

# Importa dataclass para agrupar los LLM de cada rol en un objeto inmutable
from dataclasses import dataclass
# Importa Literal para tipar las aristas condicionales
from typing import Literal

# Importa BaseChatModel para tipar los LLM
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los bloques para armar el grafo
from langgraph.graph import END, START, StateGraph

# Importa el tipo del checkpointer y del grafo compilado
from langgraph.checkpoint.base import BaseCheckpointSaver
# Importa el tipo del grafo compilado
from langgraph.graph.state import CompiledStateGraph

# Importa el nodo del analista
from app.agents.analyst_agent import crear_nodo_analista
# Importa el nodo del investigador
from app.agents.research_agent import crear_nodo_investigador
# Importa el nodo del sintetizador
from app.agents.synthesis_agent import crear_nodo_sintetizador
# Importa la configuración
from app.config import Settings
# Importa la aprobación humana y el despacho
from app.hitl import Despachador, DespachadorLog, crear_nodo_despacho, nodo_aprobacion, ruta_tras_sintesis
# Importa el estado compartido
from app.state import EstadoOrquestador
# Importa el nodo del supervisor
from app.supervisor import crear_nodo_supervisor
# Importa el nodo de validación
from app.validation import nodo_validacion


# Tipo del grafo del orquestador (estado de entrada y de salida: el estado compartido completo)
GrafoOrquestador = StateGraph[EstadoOrquestador, None, EstadoOrquestador, EstadoOrquestador]
# Tipo del grafo ya compilado (listo para ejecutar con un checkpointer)
GrafoCompilado = CompiledStateGraph[EstadoOrquestador, None, EstadoOrquestador, EstadoOrquestador]


# Un LLM por rol: permite usar un modelo más capaz para el supervisor y uno más barato para los especialistas
@dataclass(frozen=True)
class LLMs:
    # LLM del supervisor (decide el flujo)
    supervisor: BaseChatModel
    # LLM del investigador
    investigador: BaseChatModel
    # LLM del analista
    analista: BaseChatModel
    # LLM del sintetizador
    sintetizador: BaseChatModel


# Función de enrutamiento: lee la decisión del supervisor. El Literal de retorno define las aristas condicionales del diagrama
def ruta_supervisor(estado: EstadoOrquestador) -> Literal["investigador", "analista", "sintetizador"]:
    # El supervisor ya dejó en el estado un destino válido (Decision lo restringe a estos tres nombres)
    destino = estado["siguiente"]
    # Traduce el texto al Literal exacto (si llegara algo inesperado, cierra por seguridad)
    if destino == "investigador":
        return "investigador"
    if destino == "analista":
        return "analista"
    return "sintetizador"


# Arma (sin compilar) el grafo del orquestador; el despachador es quien ejecuta la acción con efectos secundarios
def crear_grafo(llms: LLMs, configuracion: Settings, despachador: Despachador | None = None) -> GrafoOrquestador:
    # Crea el grafo con el estado compartido estructurado
    grafo: GrafoOrquestador = StateGraph(EstadoOrquestador)

    # Nodo del supervisor (router inteligente y controlador de flujo)
    grafo.add_node("supervisor", crear_nodo_supervisor(llms.supervisor, configuracion))
    # Nodo del especialista de investigación
    grafo.add_node("investigador", crear_nodo_investigador(llms.investigador, configuracion))
    # Nodo del especialista de análisis
    grafo.add_node("analista", crear_nodo_analista(llms.analista, configuracion))
    # Nodo de validación (árbitro determinista)
    grafo.add_node("validacion", nodo_validacion)
    # Nodo de síntesis final
    grafo.add_node("sintetizador", crear_nodo_sintetizador(llms.sintetizador))
    # Nodo de aprobación humana: pausa el grafo (interrupt) hasta que una persona decida
    grafo.add_node("aprobacion", nodo_aprobacion)
    # Nodo de despacho: la única acción con efectos secundarios (envía el informe si fue aprobado)
    grafo.add_node("despacho", crear_nodo_despacho(despachador or DespachadorLog()))

    # El flujo siempre arranca en el supervisor
    grafo.add_edge(START, "supervisor")
    # Aristas condicionales: el supervisor decide a cuál de los tres nodos ir
    grafo.add_conditional_edges(
        "supervisor",
        ruta_supervisor,
        {"investigador": "investigador", "analista": "analista", "sintetizador": "sintetizador"},
    )
    # Cada especialista entrega su trabajo a la validación (nunca directo al supervisor ni a otro especialista)
    grafo.add_edge("investigador", "validacion")
    grafo.add_edge("analista", "validacion")
    # La validación devuelve el control al supervisor, que decide si refinar o cerrar
    grafo.add_edge("validacion", "supervisor")
    # Después de sintetizar: las tareas críticas pasan por la aprobación humana, las demás terminan
    grafo.add_conditional_edges("sintetizador", ruta_tras_sintesis, {"aprobacion": "aprobacion", "fin": END})
    # Tras la decisión humana se despacha (el nodo no envía nada si fue rechazada) y termina
    grafo.add_edge("aprobacion", "despacho")
    grafo.add_edge("despacho", END)

    # Devuelve el grafo para que el llamador lo compile
    return grafo


# Compila el orquestador con su checkpointer (Redis en producción, memoria en los tests)
# Sin checkpointer no existiría la pausa HITL: interrupt necesita guardar el estado para poder reanudarlo
def compilar_orquestador(
    llms: LLMs,
    configuracion: Settings,
    checkpointer: BaseCheckpointSaver[str],
    despachador: Despachador | None = None,
) -> GrafoCompilado:
    return crear_grafo(llms, configuracion, despachador).compile(checkpointer=checkpointer)
