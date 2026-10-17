# Módulo de observabilidad: envía a Arize Phoenix una traza por cada ejecución (cada nodo del grafo, cada llamada al LLM y cada herramienta)
# Phoenix recibe los spans por OpenTelemetry (estándar OpenInference) y calcula solo el costo (a partir de los tokens) y la latencia
#
# Dos capas de instrumentación:
#   1. Automática: LangChainInstrumentor (activado explícitamente) traza todo lo que pasa por LangChain/LangGraph (nodos, LLM, herramientas, tokens, modelo)
#   2. Manual: un span raíz por trabajo (span_de_job) con el job_id como session.id, para encontrar en el dashboard la traza de cada tarea

# Importa Iterator para tipar el generador del span
from collections.abc import Iterator
# Importa contextmanager para armar el span del trabajo con un "with"
from contextlib import contextmanager
# Importa logging para avisar si la observabilidad queda desactivada
import logging

# Importa las piezas de OpenInference para el span manual y los atributos de sesión
from openinference.instrumentation import OITracer, TraceConfig, using_attributes
# Importa el tipo del span para tipar el contexto del trabajo
from openinference.instrumentation._spans import OpenInferenceSpan
# Importa la API de OpenTelemetry para pedir el tracer global
from opentelemetry import trace as trace_api

# Importa la configuración
from app.config import Settings

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Nombre del tracer manual
NOMBRE_TRACER = "pre-entrega-7"


# Configura el log por consola (lo usan la API y el worker)
def configurar_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Oculta el log de cada petición HTTP de las librerías
    for nombre in ("httpx", "httpcore", "anthropic", "openai"):
        logging.getLogger(nombre).setLevel(logging.WARNING)


# Inicializa Phoenix: registra el exportador OTLP y activa la instrumentación automática. Devuelve el proveedor (para vaciarlo al cerrar) o None si está desactivada
def iniciar_observabilidad(configuracion: Settings) -> object | None:
    # Sin endpoint la observabilidad queda apagada (por ejemplo en los tests): el resto del sistema funciona igual
    if not configuracion.phoenix_endpoint:
        logger.info("Observabilidad desactivada (PHOENIX_COLLECTOR_ENDPOINT vacío)")
        return None
    # Se importa acá para que los tests y el modo sin Phoenix no carguen el exportador
    from openinference.instrumentation.langchain import LangChainInstrumentor
    # Importa la función que registra Phoenix como destino de las trazas
    from phoenix.otel import register

    # batch=True exporta los spans desde un hilo aparte: con el exportador "simple" cada span se enviaría por red DENTRO del event loop y lo bloquearía
    # auto_instrument=False: se instrumenta SOLO LangChain/LangGraph (abajo). Con True también se trazaría cada petición HTTP de FastAPI
    # (incluido el polling de GET /tasks/{id}) y esas trazas diminutas distorsionarían el costo y la latencia p95 del dashboard
    proveedor = register(
        project_name=configuracion.phoenix_proyecto,
        endpoint=configuracion.phoenix_endpoint,
        batch=True,
        auto_instrument=False,
        verbose=False,
    )
    # Instrumentación automática de LangChain/LangGraph: traza cada nodo, cada llamada al LLM (con modelo y tokens) y cada herramienta
    LangChainInstrumentor().instrument(tracer_provider=proveedor)
    logger.info("Observabilidad activa: trazas a %s (proyecto %s)", configuracion.phoenix_endpoint, configuracion.phoenix_proyecto)
    return proveedor


# Vacía los spans pendientes de enviar (hay que llamarlo al cerrar el proceso; si no, se pierden los últimos)
def cerrar_observabilidad(proveedor: object | None) -> None:
    flush = getattr(proveedor, "force_flush", None)
    if callable(flush):
        flush()


# Tracer manual de OpenInference (si no hay proveedor configurado, los spans son no-op y no cuestan nada)
def obtener_tracer() -> OITracer:
    return OITracer(trace_api.get_tracer(NOMBRE_TRACER), TraceConfig())


# Abre el span raíz de un trabajo: todos los spans del grafo cuelgan de él y llevan session.id = job_id
# La ejecución y su reanudación tras una aprobación son dos trazas distintas pero comparten la sesión, así que se ven juntas en Phoenix
# Si adentro se lanza una excepción el span queda en ERROR automáticamente
@contextmanager
def span_de_job(job_id: str, nombre: str, entrada: str, etiquetas: list[str] | None = None) -> Iterator[OpenInferenceSpan]:
    with using_attributes(session_id=job_id, metadata={"job_id": job_id}, tags=etiquetas or []):
        with obtener_tracer().start_as_current_span(nombre, openinference_span_kind="chain") as span:
            span.set_input(entrada)
            yield span
