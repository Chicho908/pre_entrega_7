# Agente de Análisis/Cómputo: calcula sentimiento y estadísticas sobre los datos que encontró el investigador
# Herramientas acotadas (mínimo privilegio): solo puede analizar sentimiento y calcular estadísticas; no busca información

# Importa json para devolver los resultados de las herramientas en formato estructurado
import json
# Importa logging para dejar registro cuando algo falla
import logging
# Importa Any y Literal para tipar
from typing import Any, Literal

# Importa BaseChatModel para tipar el LLM que recibe el especialista
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los tipos de mensajes que se leen para armar el resultado
from langchain_core.messages import BaseMessage, ToolMessage
# Importa BaseTool y tool para definir las herramientas
from langchain_core.tools import BaseTool, tool
# Importa BaseModel y Field para validar la entrada de cada herramienta
from pydantic import BaseModel, Field

# Importa la normalización de texto de la base de conocimiento
from app.knowledge_base import normalizar
# Importa la configuración
from app.config import Settings
# Importa el constructor y el ejecutor de agentes ReAct
from app.agents.react import ErrorAgente, Nodo, crear_agente_react, ejecutar_agente
# Importa el estado compartido
from app.state import EstadoOrquestador
# Importa utilidades de traza
from app.traza import extraer_pasos, extraer_texto

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Instrucciones fijas del rol
PROMPT_ANALISTA = (
    "Sos el Analista de un equipo de análisis de opiniones de clientes. "
    "Recibís una lista de reseñas (doc_id, rating y texto) que otro especialista ya encontró. "
    "Tu único trabajo es procesarlas con tus herramientas: "
    "1) llamá a analizar_sentimiento UNA VEZ POR CADA reseña (podés pedir varias llamadas juntas); "
    "2) llamá a calcular_estadisticas con nombre='sentimiento' y la lista de puntajes que devolvió analizar_sentimiento; "
    "3) llamá a calcular_estadisticas con nombre='rating' y la lista de ratings de las reseñas. "
    "Usá exactamente los valores recibidos y devueltos por las herramientas: nunca inventes ni redondees cifras. "
    "No busques información nueva. "
    "Cuando termines, respondé con 2 o 3 frases de conclusiones apoyadas en los números."
)

# Palabras con carga positiva (ya normalizadas, sin tildes)
PALABRAS_POSITIVAS = frozenset(
    {"excelente", "rapido", "rapida", "bueno", "buena", "buen", "perfecto", "perfecta", "recomiendo", "recomiendas",
     "recomiendan", "genial", "comodas", "comodo", "comodos", "calidad", "amable", "facil", "fantastico", "resolvio", "resolvieron"}
)
# Palabras con carga negativa
PALABRAS_NEGATIVAS = frozenset(
    {"lento", "lenta", "tarde", "tardaron", "demora", "pesima", "pesimo", "mala", "malo", "defectuoso", "decepcionante",
     "roto", "rota", "golpeada", "caras", "caro", "nunca", "problema", "problemas", "dejaron", "terrible"}
)
# Palabras que invierten el significado de la siguiente ("no es bueno")
NEGACIONES = frozenset({"no", "ni", "sin"})
# Puntaje a partir del cual un texto se considera positivo (o negativo, con signo contrario)
UMBRAL_SENTIMIENTO = 0.2


# Convierte un puntaje en etiqueta
def etiquetar(puntaje: float) -> Literal["positivo", "neutro", "negativo"]:
    # Positivo si supera el umbral
    if puntaje > UMBRAL_SENTIMIENTO:
        return "positivo"
    # Negativo si está por debajo del umbral negativo
    if puntaje < -UMBRAL_SENTIMIENTO:
        return "negativo"
    # En el medio es neutro
    return "neutro"


# Calcula el puntaje de sentimiento de un texto entre -1 y 1 con un diccionario de palabras (simple y verificable)
def puntuar_sentimiento(texto: str) -> float:
    # Pasa a minúsculas, sin tildes ni signos
    palabras = [palabra for palabra in normalizar(texto).replace(",", " ").replace(".", " ").split() if palabra]
    # Contadores de palabras positivas y negativas
    positivas = negativas = 0
    for posicion, palabra in enumerate(palabras):
        # Una negación en las 2 palabras anteriores invierte el signo ("no es bueno", "sin problemas")
        invertida = any(anterior in NEGACIONES for anterior in palabras[max(0, posicion - 2) : posicion])
        if palabra in PALABRAS_POSITIVAS:
            negativas, positivas = (negativas + 1, positivas) if invertida else (negativas, positivas + 1)
        elif palabra in PALABRAS_NEGATIVAS:
            positivas, negativas = (positivas + 1, negativas) if invertida else (positivas, negativas + 1)
    # Sin palabras con carga el texto es neutro
    total = positivas + negativas
    if total == 0:
        return 0.0
    # Balance entre -1 y 1 redondeado a 2 decimales
    return round((positivas - negativas) / total, 2)


# Esquema de entrada de analizar_sentimiento
class SentimientoInput(BaseModel):
    # Documento al que pertenece el texto
    doc_id: str = Field(min_length=2, max_length=10, description="Identificador de la reseña, por ejemplo 'd03'.")
    # Texto a analizar
    texto: str = Field(min_length=3, max_length=2000, description="Texto de la reseña, copiado tal cual.")


# Esquema de entrada de calcular_estadisticas
class EstadisticasInput(BaseModel):
    # Qué serie de datos se está resumiendo
    nombre: Literal["sentimiento", "rating"] = Field(description="'sentimiento' para los puntajes de sentimiento o 'rating' para los ratings.")
    # Los valores numéricos
    valores: list[float] = Field(min_length=1, max_length=100, description="Lista de números a resumir.")


# Herramienta 1: sentimiento de un texto
@tool(args_schema=SentimientoInput)
async def analizar_sentimiento(doc_id: str, texto: str) -> str:
    """Calcula el sentimiento de UNA reseña. Usar una vez por cada reseña a analizar.
    Devuelve un JSON con doc_id, puntaje (de -1 muy negativo a 1 muy positivo) y etiqueta (positivo, neutro o negativo)."""
    # Calcula el puntaje con el diccionario de palabras
    puntaje = puntuar_sentimiento(texto)
    # Devuelve el resultado estructurado
    return json.dumps({"doc_id": doc_id, "puntaje": puntaje, "etiqueta": etiquetar(puntaje)}, ensure_ascii=False)


# Herramienta 2: estadísticas de una lista de números
@tool(args_schema=EstadisticasInput)
async def calcular_estadisticas(nombre: str, valores: list[float]) -> str:
    """Calcula cantidad, promedio, mínimo y máximo de una lista de números.
    Usar con nombre='sentimiento' (puntajes de analizar_sentimiento) y con nombre='rating' (ratings de las reseñas).
    Devuelve un JSON con nombre, cantidad, promedio, minimo y maximo."""
    # Calcula las estadísticas básicas
    resultado = {
        "nombre": nombre,
        "cantidad": len(valores),
        "promedio": round(sum(valores) / len(valores), 2),
        "minimo": min(valores),
        "maximo": max(valores),
    }
    # Devuelve el resultado estructurado
    return json.dumps(resultado, ensure_ascii=False)


# Lista de herramientas del analista
HERRAMIENTAS_ANALISTA: list[BaseTool] = [analizar_sentimiento, calcular_estadisticas]


# Arma la instrucción del analista: la orden del supervisor + SOLO los datos que necesita (nada más del estado)
def armar_instruccion_analista(instruccion: str, investigacion: dict[str, Any]) -> str:
    # Una línea por reseña: id, rating y texto
    filas = [f"- {h['doc_id']} | rating {h['rating']} | {h['texto']}" for h in investigacion.get("hallazgos", [])]
    # Une la orden con los datos
    return f"{instruccion}\n\nReseñas a analizar:\n" + "\n".join(filas)


# Elige qué estadística informar cuando hay varias de la misma serie: la que cubre todas las reseñas (global); si no hay, la última
def _elegir_estadistica(candidatas: list[dict[str, Any]], total: int) -> dict[str, Any] | None:
    if not candidatas:
        return None
    globales = [c for c in candidatas if c.get("cantidad") == total]
    return globales[-1] if globales else candidatas[-1]


# Arma el resultado del análisis SOLO con lo que devolvieron las herramientas (los números no los inventa el LLM)
def construir_analisis(mensajes: list[BaseMessage]) -> dict[str, Any]:
    # Sentimientos por doc_id
    sentimientos: dict[str, dict[str, Any]] = {}
    # Estadísticas por nombre de serie (puede haber varias: por ejemplo una por producto en una comparación, más la global)
    estadisticas: dict[str, list[dict[str, Any]]] = {}
    # Recorre los resultados de herramientas
    for mensaje in mensajes:
        # Ignora todo lo que no sea el resultado de una herramienta
        if not isinstance(mensaje, ToolMessage):
            continue
        # Lee el texto devuelto
        contenido = extraer_texto(mensaje)
        # Ignora los errores
        if contenido.startswith("ERROR"):
            continue
        # Convierte el JSON en datos
        try:
            datos = json.loads(contenido)
        except json.JSONDecodeError:
            continue
        # Clasifica el resultado según qué herramienta lo generó
        if mensaje.name == "analizar_sentimiento":
            sentimientos[datos["doc_id"]] = datos
        elif mensaje.name == "calcular_estadisticas":
            estadisticas.setdefault(datos["nombre"], []).append(datos)
    # Arma el diccionario solo con los datos disponibles: si falta algo, la validación lo va a señalar
    analisis: dict[str, Any] = {
        "sentimientos": list(sentimientos.values()),
        "cantidad_analizada": len(sentimientos),
        "conclusiones": extraer_texto(mensajes[-1]).strip() if mensajes else "",
    }
    # Promedio de sentimiento (si el analista lo calculó) y su etiqueta global
    sentimiento = _elegir_estadistica(estadisticas.get("sentimiento", []), len(sentimientos))
    if sentimiento is not None:
        analisis["puntaje_promedio"] = sentimiento["promedio"]
        analisis["sentimiento_global"] = etiquetar(sentimiento["promedio"])
    # Promedio de rating (si el analista lo calculó)
    rating = _elegir_estadistica(estadisticas.get("rating", []), len(sentimientos))
    if rating is not None:
        analisis["rating_promedio"] = rating["promedio"]
    return analisis


# Crea el nodo del analista para el grafo principal
def crear_nodo_analista(llm: BaseChatModel, configuracion: Settings) -> Nodo:
    # Crea el agente ReAct del analista
    agente = crear_agente_react(llm, HERRAMIENTAS_ANALISTA, PROMPT_ANALISTA)

    # Nodo: recibe el estado, trabaja con la instrucción + los hallazgos y devuelve solo sus campos
    async def nodo_analista(state: EstadoOrquestador) -> dict[str, Any]:
        # Cuenta este intento
        intento = state.get("intentos_analista", 0) + 1
        # Arma lo que ve el analista: la orden y los datos (no ve la conversación ni las decisiones del supervisor)
        instruccion = armar_instruccion_analista(state["instruccion"], state.get("investigacion", {}))
        # Intenta correr al especialista
        try:
            mensajes = await ejecutar_agente(agente, instruccion, configuracion.limite_iteraciones_agente)
            analisis = construir_analisis(mensajes)
            pasos = extraer_pasos(mensajes[1:])
            error = ""
            resumen = f"analizó {analisis['cantidad_analizada']} reseña(s)"
        # Si falla, lo registra y deja el análisis vacío (la validación lo va a rechazar)
        except ErrorAgente as falla:
            logger.warning("El analista falló: %s", falla)
            analisis, pasos, error, resumen = {}, [], str(falla), "no pudo completar el análisis"
        # Devuelve solo sus campos + su aporte al registro de contribuciones
        return {
            "analisis": analisis,
            "intentos_analista": intento,
            "ultimo_agente": "analista",
            "contribuciones": [{"agente": "analista", "intento": intento, "resumen": resumen, "pasos": pasos, "error": error}],
        }

    return nodo_analista
