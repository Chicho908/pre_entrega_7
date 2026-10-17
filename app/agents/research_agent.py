# Agente de Investigación: consulta la base de conocimiento de reseñas y devuelve hallazgos con su procedencia
# Herramientas acotadas (mínimo privilegio): solo puede BUSCAR reseñas; no calcula ni analiza nada

# Importa json para devolver los resultados de las herramientas en formato estructurado
import json
# Importa logging para dejar registro cuando algo falla
import logging
# Importa Any para tipar los diccionarios de estado
from typing import Any

# Importa BaseChatModel para tipar el LLM que recibe el especialista
from langchain_core.language_models.chat_models import BaseChatModel
# Importa los tipos de mensajes que se leen para armar el resultado
from langchain_core.messages import BaseMessage, ToolMessage
# Importa BaseTool y tool para definir las herramientas
from langchain_core.tools import BaseTool, ToolException, tool
# Importa BaseModel y Field para validar la entrada de cada herramienta
from pydantic import BaseModel, Field

# Importa la base de conocimiento simulada
from app import knowledge_base
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

# Prefijo con el que se reconoce una falla de infraestructura de la fuente (distinta de "no hay datos")
PREFIJO_FUENTE_CAIDA = "FUENTE_CAIDA"

# Instrucciones fijas del rol (el LLM del investigador solo conoce este rol, no el resto del sistema)
PROMPT_INVESTIGADOR = (
    "Sos el Investigador de un equipo de análisis de opiniones de clientes. "
    "Tu único trabajo es encontrar reseñas relevantes usando tu herramienta buscar_resenas (búsqueda por palabras clave). "
    "Nunca inventes reseñas ni cifras: solo podés informar lo que las herramientas devuelvan. "
    "Si una búsqueda devuelve estado 'sin_resultados', probá una vez más con sinónimos o con menos palabras, pero SIEMPRE sobre el mismo producto o tema que te pidieron. "
    "Si la herramienta falla con FUENTE_CAIDA, es una falla técnica (no falta de datos): reintentá UNA vez la misma búsqueda y, si vuelve a fallar, terminá sin reseñas y explicá que la fuente no respondió. "
    "Si el producto o tema pedido no existe en la base, NO lo reemplaces por términos genéricos (como 'calidad' o 'precio') solo para obtener resultados: "
    "terminá sin reseñas y explicá que no hay datos sobre lo pedido. "
    "No analices sentimiento ni calcules promedios: eso lo hace otro especialista. "
    "Cuando termines, respondé con un resumen de 1 o 2 frases de lo que encontraste (cuántas reseñas y de qué trata cada tema)."
)


# Esquema de entrada de buscar_resenas
class BuscarInput(BaseModel):
    # Palabras clave de la búsqueda
    consulta: str = Field(min_length=2, max_length=200, description="Palabras clave, por ejemplo 'entrega auriculares'.")
    # Cantidad máxima de reseñas a devolver
    max_resultados: int = Field(default=5, ge=1, le=10, description="Cantidad máxima de reseñas (1 a 10).")


# Convierte un documento de la base en un diccionario serializable
def _a_dict(documento: knowledge_base.Documento) -> dict[str, Any]:
    return {
        "doc_id": documento.id,
        "producto": documento.producto,
        "categoria": documento.categoria,
        "rating": documento.rating,
        "texto": documento.texto,
    }


# Herramienta del investigador: búsqueda por palabras clave
@tool(args_schema=BuscarInput)
async def buscar_resenas(consulta: str, max_resultados: int = 5) -> str:
    """Busca reseñas de clientes por palabras clave (producto y/o tema como entrega, calidad, precio o soporte).
    Solo devuelve reseñas que contengan TODAS las palabras de la consulta, así que conviene usar pocas palabras
    (por ejemplo 'entrega auriculares'). Devuelve un JSON con estado ('ok' o 'sin_resultados') y la lista de reseñas
    (doc_id, producto, categoria, rating y texto). Si la fuente no responde, falla con un error FUENTE_CAIDA."""
    # Intenta consultar la fuente
    try:
        documentos = await knowledge_base.buscar(consulta, max_resultados)
    # Falla de infraestructura: se lanza como excepción (no como texto) para que el span de la herramienta quede en ERROR en el dashboard
    except Exception as error:
        logger.exception("Falla al buscar '%s'", consulta)
        raise ToolException(f"{PREFIJO_FUENTE_CAIDA}: la fuente de reseñas no respondió ({type(error).__name__}). Es una falla técnica, no falta de datos.") from error
    # Sin resultados NO es un error: la fuente funcionó y simplemente no hay datos (el span queda en OK)
    if not documentos:
        return json.dumps(
            {
                "consulta": consulta,
                "estado": "sin_resultados",
                "resultados": [],
                "mensaje": (
                    f"Ninguna reseña coincide con todas las palabras de '{consulta}'. Probá con menos palabras o con sinónimos del mismo producto o tema. "
                    "La base solo cubre estos productos: auriculares, zapatillas, cafetera, teclado; y estos temas: entrega, calidad, precio, soporte. "
                    "Si lo pedido no está en esa lista, no hay datos: no lo reemplaces por otro tema."
                ),
            },
            ensure_ascii=False,
        )
    # Devuelve los resultados en JSON
    return json.dumps({"consulta": consulta, "estado": "ok", "resultados": [_a_dict(d) for d in documentos]}, ensure_ascii=False)


# Lista de herramientas del investigador
HERRAMIENTAS_INVESTIGADOR: list[BaseTool] = [buscar_resenas]


# Arma el resultado de la investigación SOLO con lo que devolvieron las herramientas (el LLM no puede inventar hallazgos)
def construir_investigacion(instruccion: str, mensajes: list[BaseMessage]) -> dict[str, Any]:
    # Hallazgos por doc_id (si dos búsquedas devuelven el mismo documento, se guarda una vez)
    hallazgos: dict[str, dict[str, Any]] = {}
    # Marca si alguna búsqueda falló por la infraestructura de la fuente
    fuente_caida = False
    # Recorre los resultados de herramientas
    for mensaje in mensajes:
        # Ignora todo lo que no sea el resultado de una herramienta
        if not isinstance(mensaje, ToolMessage):
            continue
        # Lee el texto devuelto por la herramienta
        contenido = extraer_texto(mensaje)
        # Una herramienta que falló no aporta hallazgos; si fue por la fuente, queda registrado
        if mensaje.status == "error":
            fuente_caida = fuente_caida or contenido.startswith(PREFIJO_FUENTE_CAIDA)
            continue
        # Convierte el JSON en datos
        try:
            datos = json.loads(contenido)
        except json.JSONDecodeError:
            continue
        # Guarda cada documento encontrado
        for resultado in datos.get("resultados", []):
            hallazgos[resultado["doc_id"]] = resultado
    # El resumen es la última respuesta de texto del LLM
    resumen = extraer_texto(mensajes[-1]).strip() if mensajes else ""
    # Devuelve el diccionario crudo: la validación decide después si cumple el contrato
    return {"consulta": instruccion, "hallazgos": list(hallazgos.values()), "resumen": resumen, "fuente_caida": fuente_caida}


# Crea el nodo del investigador para el grafo principal
def crear_nodo_investigador(llm: BaseChatModel, configuracion: Settings) -> Nodo:
    # Crea el agente ReAct del investigador (una sola vez, no en cada ejecución)
    agente = crear_agente_react(llm, HERRAMIENTAS_INVESTIGADOR, PROMPT_INVESTIGADOR)

    # Nodo: recibe el estado, trabaja con SOLO la instrucción y devuelve únicamente los campos que le pertenecen
    async def nodo_investigador(state: EstadoOrquestador) -> dict[str, Any]:
        # Cuenta este intento
        intento = state.get("intentos_investigador", 0) + 1
        # La instrucción específica que dejó el supervisor (el investigador NO ve el historial de la conversación)
        instruccion = state["instruccion"]
        # Intenta correr al especialista
        try:
            mensajes = await ejecutar_agente(agente, instruccion, configuracion.limite_iteraciones_agente)
            investigacion = construir_investigacion(instruccion, mensajes)
            pasos = extraer_pasos(mensajes[1:])
            error = ""
            resumen = f"encontró {len(investigacion['hallazgos'])} reseña(s)"
            if investigacion["fuente_caida"] and not investigacion["hallazgos"]:
                resumen = "la fuente de reseñas no respondió"
        # Si falla, lo registra y deja la investigación vacía (la validación la va a rechazar)
        except ErrorAgente as falla:
            logger.warning("El investigador falló: %s", falla)
            investigacion, pasos, error, resumen = {}, [], str(falla), "no pudo completar la búsqueda"
        # Devuelve solo sus campos + su aporte al registro de contribuciones
        return {
            "investigacion": investigacion,
            "intentos_investigador": intento,
            "ultimo_agente": "investigador",
            "contribuciones": [{"agente": "investigador", "intento": intento, "resumen": resumen, "pasos": pasos, "error": error}],
        }

    return nodo_investigador
