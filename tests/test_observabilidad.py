# Tests de la observabilidad: se capturan los spans en memoria (sin Phoenix) y se comprueba lo que el dashboard mostraría
#   - una falla de infraestructura deja el span de la herramienta en ERROR; "sin resultados" lo deja en OK
#   - cada trabajo tiene un span raíz con session.id = job_id, y los spans del grafo cuelgan de él

# Importa Iterator para tipar la fixture
from collections.abc import Iterator

# Importa pytest
import pytest
# Importa el instrumentador de LangChain (el mismo que activa phoenix.otel.register(auto_instrument=True))
from openinference.instrumentation.langchain import LangChainInstrumentor
# Importa la API de OpenTelemetry para leer el span activo
from opentelemetry import trace as trace_api
# Importa el proveedor de trazas y el tipo de span del SDK
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
# Importa el procesador que entrega cada span al exportador
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
# Importa el exportador que guarda los spans en memoria
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
# Importa los códigos de estado de un span
from opentelemetry.trace import StatusCode
# Importa AIMessage para armar las respuestas falsas del LLM
from langchain_core.messages import AIMessage

# Importa lo que se prueba
from app import knowledge_base
# Importa la configuración
from app.config import Settings
# Importa la inicialización de Phoenix y el span de cada trabajo
from app.observability import iniciar_observabilidad, span_de_job
# Importa el estado inicial del orquestador
from app.state import estado_inicial
# Importa los dobles de prueba
from tests.fakes import armar_orquestador, decidir, guion_sintetizador, pedir, settings_de_prueba

# Exportador y proveedor compartidos (OpenTelemetry solo permite fijar el proveedor global una vez por proceso)
EXPORTADOR = InMemorySpanExporter()
PROVEEDOR = TracerProvider()
PROVEEDOR.add_span_processor(SimpleSpanProcessor(EXPORTADOR))
trace_api.set_tracer_provider(PROVEEDOR)


# Instrumenta LangChain durante cada test y limpia los spans
@pytest.fixture(autouse=True)
def instrumentar() -> Iterator[None]:
    EXPORTADOR.clear()
    LangChainInstrumentor().instrument(tracer_provider=PROVEEDOR, skip_dep_check=True)
    yield
    LangChainInstrumentor().uninstrument()


# Devuelve los spans de herramienta con ese nombre
def spans_de_herramienta(nombre: str) -> list[ReadableSpan]:
    return [s for s in EXPORTADOR.get_finished_spans() if s.name == nombre and (s.attributes or {}).get("openinference.span.kind") == "TOOL"]


# Corre un orquestador cuyo investigador busca una vez
async def correr_busqueda(consulta: str) -> None:
    investigador = [pedir(("buscar_resenas", {"consulta": consulta}), prefijo="i"), AIMessage(content="Listo.")]
    grafo, _ = armar_orquestador(
        [decidir("investigador"), decidir("sintetizador")], investigador, [], guion_sintetizador(), configuracion=settings_de_prueba(max_refinamientos=0)
    )
    await grafo.ainvoke(estado_inicial("pregunta de prueba"), config={"recursion_limit": 60})


# Test 1: la fuente caída deja el span de la herramienta en ERROR
async def test_fuente_caida_marca_el_span_en_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fuente de datos que falla como si la base estuviera caída
    async def buscar_roto(consulta: str, max_resultados: int = 5) -> list[knowledge_base.Documento]:
        raise ConnectionError("la base no responde")

    monkeypatch.setattr(knowledge_base, "buscar", buscar_roto)
    await correr_busqueda("entrega auriculares")
    spans = spans_de_herramienta("buscar_resenas")
    assert len(spans) == 1 and spans[0].status.status_code == StatusCode.ERROR


# Test 2: "sin resultados" es un resultado válido de la herramienta: el span queda en OK (no es una falla)
async def test_sin_resultados_no_marca_error() -> None:
    await correr_busqueda("drones")
    spans = spans_de_herramienta("buscar_resenas")
    assert len(spans) == 1 and spans[0].status.status_code != StatusCode.ERROR


# Test 3: una búsqueda con datos tampoco es error
async def test_busqueda_con_datos_no_marca_error() -> None:
    await correr_busqueda("entrega auriculares")
    assert spans_de_herramienta("buscar_resenas")[0].status.status_code != StatusCode.ERROR


# Test 4: el span raíz del trabajo lleva el job_id como sesión, y los spans del grafo cuelgan de él con la misma sesión
async def test_span_de_job_agrupa_la_traza_por_sesion() -> None:
    with span_de_job("job-123", "job.ejecutar", "pregunta de prueba", etiquetas=["ejecucion"]) as span:
        await correr_busqueda("entrega auriculares")
        span.set_output("respuesta")
    spans = EXPORTADOR.get_finished_spans()
    raiz = next(s for s in spans if s.name == "job.ejecutar")
    assert (raiz.attributes or {})["session.id"] == "job-123" and raiz.parent is None
    # Todos los spans del grafo pertenecen a la misma traza y heredan la sesión
    resto = [s for s in spans if s is not raiz]
    assert resto and all(s.context is not None and raiz.context is not None and s.context.trace_id == raiz.context.trace_id for s in resto)
    assert all((s.attributes or {}).get("session.id") == "job-123" for s in resto if (s.attributes or {}).get("openinference.span.kind") == "LLM")


# Test 5: si el trabajo lanza una excepción, el span raíz queda en ERROR
def test_span_de_job_queda_en_error_si_falla() -> None:
    with pytest.raises(RuntimeError):
        with span_de_job("job-err", "job.ejecutar", "x"):
            raise RuntimeError("falló")
    raiz = next(s for s in EXPORTADOR.get_finished_spans() if s.name == "job.ejecutar")
    assert raiz.status.status_code == StatusCode.ERROR


# Test 6: sin PHOENIX_COLLECTOR_ENDPOINT la observabilidad queda desactivada (y el sistema funciona igual)
def test_observabilidad_desactivada_sin_endpoint() -> None:
    ajustes: Settings = settings_de_prueba(phoenix_endpoint="")
    assert iniciar_observabilidad(ajustes) is None
