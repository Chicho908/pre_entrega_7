# Módulo del supervisor: el "cerebro" que decide qué nodo actúa ahora
# Dos capas: (1) un LLM propone el próximo paso con salida estructurada y (2) guardarraíles de código lo corrigen si rompe las reglas
# Así se combina la flexibilidad del LLM con garantías duras: nunca hay un supervisor infinito ni se cierra sin una rúbrica cumplida

# Importa logging para dejar registro de las decisiones y de las correcciones
import logging
# Importa Mapping para tipar el estado de solo lectura
from collections.abc import Mapping
# Importa Any para tipar los diccionarios de estado
from typing import Any

# Importa BaseChatModel para tipar el LLM del supervisor
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los mensajes para armar la consulta al LLM
from langchain_core.messages import HumanMessage, SystemMessage

# Importa el tipo de nodo
from app.agents.react import Nodo
# Importa la pregunta original
from app.agents.synthesis_agent import pregunta_original
# Importa la configuración
from app.config import Settings
# Importa el detector de errores de credenciales del LLM
from app.llm import es_error_de_credenciales
# Importa los contratos
from app.state import Decision, Destino, EstadoOrquestador

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Prompt del supervisor, con la rúbrica de validación personalizada
PROMPT_SUPERVISOR = """Sos el Supervisor de un equipo que responde preguntas sobre opiniones de clientes.
Tu equipo:
- investigador: busca reseñas relevantes en la base de conocimiento.
- analista: calcula sentimiento y estadísticas sobre las reseñas que encontró el investigador.
- sintetizador: redacta la respuesta final. Es el cierre: después de él no se puede corregir nada.

En cada turno recibís el ESTADO ACTUAL (resumen). Dada la situación, ¿quién debe intervenir ahora o es momento de finalizar?
Elegí en "siguiente" exactamente uno de: investigador, analista, sintetizador.

Orden normal: investigador → analista → sintetizador.

RÚBRICA DE SUFICIENCIA (solo elegí sintetizador si se cumple TODO):
1. La investigación está VALIDADA y tiene al menos una reseña.
2. El análisis está VALIDADO y cubre todas las reseñas encontradas.
3. La pregunta del usuario queda respondida con esos datos.
Si un especialista fue RECHAZADO por la validación, volvé a llamarlo con una instrucción que corrija el error específico que se informa.
Al reintentar, NO cambies el tema que pidió el usuario por temas genéricos: se pueden probar sinónimos del mismo producto o tema, pero si lo pedido no existe en la base, el reintento no debe inventar otra búsqueda.
Si el presupuesto de intentos de un especialista se agotó, elegí sintetizador: la respuesta final va a explicar la limitación.

La "instruccion" tiene que ser AUTOCONTENIDA: el especialista NO ve la conversación ni este estado. Incluí qué buscar o calcular y sobre qué.
Para el sintetizador dejá la instruccion vacía. En "razon" explicá en una frase por qué elegiste ese paso."""


# Arma el resumen del estado que ve el supervisor (no el historial completo: evita la contaminación de contexto)
def resumir_estado(estado: Mapping[str, Any], paso: int, max_pasos: int, max_intentos: int) -> str:
    lineas = [
        f"Pregunta del usuario: {pregunta_original(estado)}",
        f"Paso del supervisor: {paso} de {max_pasos} (al llegar al tope se fuerza el cierre).",
        "",
    ]
    # Estado de la investigación
    if estado.get("intentos_investigador", 0) == 0:
        lineas.append("Investigación: NO realizada.")
    elif estado.get("investigacion_valida"):
        lineas.append(f"Investigación: VALIDADA ({len(estado['investigacion']['hallazgos'])} reseñas).")
    else:
        lineas.append(f"Investigación: RECHAZADA por la validación → {estado.get('error_investigacion')}")
    lineas.append(f"Intentos del investigador: {estado.get('intentos_investigador', 0)} de {max_intentos}.")
    # Estado del análisis
    if estado.get("intentos_analista", 0) == 0:
        lineas.append("Análisis: NO realizado.")
    elif estado.get("analisis_valido"):
        lineas.append("Análisis: VALIDADO.")
    else:
        lineas.append(f"Análisis: RECHAZADO por la validación → {estado.get('error_analisis')}")
    lineas.append(f"Intentos del analista: {estado.get('intentos_analista', 0)} de {max_intentos}.")
    return "\n".join(lineas)


# Decisión determinista según el estado (se usa como respaldo si el LLM falla)
def decision_por_defecto(estado: Mapping[str, Any], max_intentos: int) -> Decision:
    # Sin investigación válida y con intentos disponibles: investigar
    if not estado.get("investigacion_valida") and estado.get("intentos_investigador", 0) < max_intentos:
        return Decision(siguiente="investigador", instruccion="", razon="regla por defecto: falta una investigación válida")
    # Con investigación válida pero sin análisis válido y con intentos disponibles: analizar
    if estado.get("investigacion_valida") and not estado.get("analisis_valido") and estado.get("intentos_analista", 0) < max_intentos:
        return Decision(siguiente="analista", instruccion="", razon="regla por defecto: falta un análisis válido")
    # En cualquier otro caso: cerrar
    return Decision(siguiente="sintetizador", instruccion="", razon="regla por defecto: no hay más pasos útiles")


# Instrucción por defecto de un especialista (si el LLM no dio una o hubo que redirigirlo)
def instruccion_por_defecto(destino: str, estado: Mapping[str, Any]) -> str:
    # El investigador recibe la pregunta original
    if destino == "investigador":
        return f"Encontrá las reseñas relevantes para responder esta pregunta: {pregunta_original(estado)}"
    # El analista recibe una orden fija (los datos se agregan en su nodo)
    if destino == "analista":
        return "Analizá el sentimiento de cada reseña y calculá los promedios de sentimiento y de rating."
    # El sintetizador no necesita instrucción
    return ""


# Aplica los guardarraíles: corrige la propuesta del LLM si rompe una regla. Devuelve (decisión final, motivo de la corrección o "")
def aplicar_reglas(propuesta: Decision, estado: Mapping[str, Any], max_intentos: int) -> tuple[Decision, str]:
    destino: Destino = propuesta.siguiente
    nota = ""
    # Contadores de intentos y validez actuales
    intentos = {"investigador": estado.get("intentos_investigador", 0), "analista": estado.get("intentos_analista", 0)}
    inv_ok = bool(estado.get("investigacion_valida"))
    ana_ok = bool(estado.get("analisis_valido"))

    # Regla 1: no se llama a un especialista que ya agotó sus intentos
    if destino in intentos and intentos[destino] >= max_intentos:
        # Si ya hay investigación válida y falta analizar, sigue con el análisis; si no, cierra
        destino = "analista" if (destino == "investigador" and inv_ok and not ana_ok and intentos["analista"] < max_intentos) else "sintetizador"
        nota = f"se agotaron los intentos del especialista elegido ({max_intentos}); se redirige a {destino}"
    # Regla 2: no se analiza sin una investigación válida
    if destino == "analista" and not inv_ok:
        destino = "investigador" if intentos["investigador"] < max_intentos else "sintetizador"
        nota = f"no se puede analizar sin una investigación validada; se redirige a {destino}"
    # Regla 3: no se re-investiga si la investigación ya es válida y el análisis aún no se hizo (se sigue el orden normal)
    if destino == "investigador" and inv_ok and not ana_ok and intentos["analista"] == 0:
        destino = "analista"
        nota = "la investigación ya está validada; se sigue con el análisis"
    # Regla 4: no se cierra si la rúbrica no se cumple y todavía hay un especialista con intentos disponibles
    if destino == "sintetizador":
        if not inv_ok and intentos["investigador"] < max_intentos:
            destino, nota = "investigador", "la rúbrica no se cumple: falta una investigación validada"
        elif inv_ok and not ana_ok and intentos["analista"] < max_intentos:
            destino, nota = "analista", "la rúbrica no se cumple: falta un análisis validado"

    # Si la propuesta no cambió, la deja intacta
    if destino == propuesta.siguiente:
        final = propuesta
    else:
        final = Decision(siguiente=destino, instruccion="", razon=propuesta.razon)
    # Completa la instrucción si falta y agrega el error previo cuando se repite un especialista
    instruccion = final.instruccion.strip() or instruccion_por_defecto(final.siguiente, estado)
    error_previo = {"investigador": estado.get("error_investigacion", ""), "analista": estado.get("error_analisis", "")}.get(final.siguiente, "")
    if error_previo and intentos.get(final.siguiente, 0) > 0 and error_previo not in instruccion:
        instruccion += f"\nCorregí este problema detectado por la validación en tu intento anterior: {error_previo}"
    # El sintetizador nunca lleva instrucción
    if final.siguiente == "sintetizador":
        instruccion = ""
    return final.model_copy(update={"instruccion": instruccion}), nota


# Crea el nodo supervisor para el grafo principal
def crear_nodo_supervisor(llm: BaseChatModel, configuracion: Settings) -> Nodo:
    # Intentos máximos por especialista: el primero + los refinamientos permitidos
    max_intentos = 1 + configuracion.max_refinamientos
    # LLM con salida estructurada: está obligado a devolver una Decision válida (el Literal impide nodos inventados)
    llm_decisor = llm.with_structured_output(Decision)

    # Nodo: decide el próximo paso y deja la decisión en el estado
    async def nodo_supervisor(state: EstadoOrquestador) -> dict[str, Any]:
        # Cuenta esta decisión
        paso = state.get("pasos_supervisor", 0) + 1
        propuesta_llm = "(sin consultar al LLM)"
        # Guardarraíl anti "supervisor infinito": al llegar al tope se cierra sin consultar al LLM
        if paso >= configuracion.max_pasos_supervisor:
            final = Decision(siguiente="sintetizador", instruccion="", razon="tope de pasos del supervisor alcanzado; se cierra con lo disponible")
            nota = f"tope de {configuracion.max_pasos_supervisor} pasos alcanzado"
        else:
            # Consulta al LLM con el resumen del estado (no con todo el historial)
            try:
                resumen = resumir_estado(state, paso, configuracion.max_pasos_supervisor, max_intentos)
                salida = await llm_decisor.ainvoke([SystemMessage(content=PROMPT_SUPERVISOR), HumanMessage(content=resumen)])
                propuesta = salida if isinstance(salida, Decision) else Decision.model_validate(salida)
                propuesta_llm = propuesta.siguiente
            # Si el LLM falla o devuelve algo inválido, decide la regla por defecto (el sistema no se cae)
            except Exception as error:
                # API key inválida: no se sigue con la regla por defecto, el trabajo debe terminar en FAILED
                if es_error_de_credenciales(error):
                    raise
                logger.exception("El LLM del supervisor falló; se usa la decisión por defecto")
                propuesta = decision_por_defecto(state, max_intentos)
                propuesta_llm = "(falló el LLM)"
            # Aplica los guardarraíles sobre la propuesta
            final, nota = aplicar_reglas(propuesta, state, max_intentos)
        # Registra la decisión para auditoría y traza
        registro = {
            "paso": paso,
            "propuesta": propuesta_llm,
            "final": final.siguiente,
            "razon": final.razon,
            "guardarrail": nota,
            "instruccion": final.instruccion,
        }
        logger.debug("Supervisor paso %s: %s → %s", paso, propuesta_llm, final.siguiente)
        # Devuelve solo los campos del supervisor
        return {"siguiente": final.siguiente, "instruccion": final.instruccion, "pasos_supervisor": paso, "decisiones": [registro]}

    return nodo_supervisor
