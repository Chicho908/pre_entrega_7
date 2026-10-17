# Agente de Síntesis: redacta la respuesta final usando SOLO los datos ya validados
# No tiene herramientas: no busca ni calcula, solo redacta (y avisa las limitaciones si algo no se pudo validar)

# Importa json para pasarle los datos al LLM en formato compacto
import json
# Importa logging para dejar registro cuando falla el LLM
import logging
# Importa Mapping para tipar el estado de solo lectura
from collections.abc import Mapping
# Importa Any para tipar los diccionarios de estado
from typing import Any

# Importa BaseChatModel para tipar el LLM del sintetizador
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los mensajes para armar el prompt y devolver la respuesta
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Importa el tipo de nodo
from app.agents.react import Nodo
# Importa el detector de errores de credenciales del LLM
from app.llm import es_error_de_credenciales
# Importa el estado compartido
from app.state import EstadoOrquestador
# Importa utilidades de traza
from app.traza import extraer_texto

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Instrucciones fijas del rol
PROMPT_SINTETIZADOR = (
    "Sos el Sintetizador de un equipo de análisis de opiniones de clientes. "
    "Redactá la respuesta final para el usuario usando EXCLUSIVAMENTE los datos validados que te pasan. "
    "No inventes cifras ni reseñas. Citá los números tal cual (promedio de rating, puntaje de sentimiento, cantidad de reseñas). "
    "La cantidad de reseñas analizadas es el campo cantidad_resenas: no cuentes por tu cuenta. "
    "Si hay LIMITACIONES, explicalas con claridad y no completes lo que falta. "
    "Respondé en español, en un párrafo corto."
)


# Obtiene la pregunta original (el primer mensaje humano del historial)
def pregunta_original(estado: Mapping[str, Any]) -> str:
    # Recorre el historial buscando el primer mensaje del usuario
    for mensaje in estado.get("messages", []):
        if isinstance(mensaje, HumanMessage):
            return extraer_texto(mensaje)
    return ""


# Arma las limitaciones: qué partes no se pudieron validar y por qué
def armar_limitaciones(estado: Mapping[str, Any]) -> list[str]:
    limitaciones: list[str] = []
    # Investigación no válida
    if not estado.get("investigacion_valida"):
        limitaciones.append(f"No se pudo validar la investigación: {estado.get('error_investigacion') or 'no se ejecutó'}.")
    # Análisis no válido
    if not estado.get("analisis_valido"):
        limitaciones.append(f"No se pudo validar el análisis: {estado.get('error_analisis') or 'no se ejecutó'}.")
    return limitaciones


# Respuesta de emergencia sin LLM: se usa si el LLM del sintetizador falla (el usuario igual recibe algo útil y honesto)
def respuesta_de_respaldo(estado: Mapping[str, Any], limitaciones: list[str]) -> str:
    partes: list[str] = []
    # Si el análisis es válido, informa los números
    if estado.get("analisis_valido"):
        analisis = estado["analisis"]
        partes.append(
            f"Se analizaron {analisis['cantidad_analizada']} reseñas: sentimiento {analisis['sentimiento_global']} "
            f"(puntaje {analisis['puntaje_promedio']}) y rating promedio {analisis['rating_promedio']}."
        )
    # Siempre informa las limitaciones
    partes.extend(limitaciones)
    return " ".join(partes) or "No se pudo generar una respuesta."


# Crea el nodo del sintetizador para el grafo principal
def crear_nodo_sintetizador(llm: BaseChatModel) -> Nodo:
    # Nodo: arma los datos validados, redacta y devuelve solo sus campos
    async def nodo_sintetizador(state: EstadoOrquestador) -> dict[str, Any]:
        # Calcula qué quedó sin validar
        limitaciones = armar_limitaciones(state)
        # Solo pasa al LLM los datos que sí pasaron la validación
        datos: dict[str, Any] = {"pregunta": pregunta_original(state), "limitaciones": limitaciones}
        if state.get("investigacion_valida"):
            # Solo los hallazgos reales: el resumen en texto libre del investigador se descarta para que no contradiga a los datos
            datos["cantidad_resenas"] = len(state["investigacion"]["hallazgos"])
            datos["hallazgos"] = state["investigacion"]["hallazgos"]
        if state.get("analisis_valido"):
            datos["analisis"] = state["analisis"]
        # Intenta redactar con el LLM
        try:
            respuesta = await llm.ainvoke([SystemMessage(content=PROMPT_SINTETIZADOR), HumanMessage(content=json.dumps(datos, ensure_ascii=False))])
            texto = extraer_texto(respuesta).strip()
        # Si el LLM falla, usa la respuesta de respaldo en vez de romper todo el flujo
        except Exception as error:
            # API key inválida: no se disimula con una respuesta de respaldo, el trabajo debe terminar en FAILED
            if es_error_de_credenciales(error):
                raise
            logger.exception("Falla del LLM en la síntesis; se usa la respuesta de respaldo")
            texto = ""
        # Si el texto quedó vacío (falla o respuesta vacía), usa la respuesta de respaldo
        texto = texto or respuesta_de_respaldo(state, limitaciones)
        # Devuelve su respuesta como mensaje del historial y como campo propio
        return {
            "messages": [AIMessage(content=texto)],
            "respuesta_final": texto,
            "contribuciones": [{"agente": "sintetizador", "intento": 1, "resumen": "redactó la respuesta final", "pasos": [], "error": ""}],
        }

    return nodo_sintetizador
