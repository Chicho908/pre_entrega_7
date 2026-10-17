# API de producción del orquestador multi-agente (FastAPI)
#
#   POST /tasks               → guarda el trabajo en Redis, lo encola y responde YA con un job_id (202). No espera al LLM
#   GET  /tasks/{id}          → estado del trabajo: PENDING / RUNNING / WAITING_APPROVAL / DONE / FAILED (y el resultado o el error)
#   POST /tasks/{id}/approve  → decisión humana para una tarea crítica pausada (ver hitl.py)
#   GET  /health              → comprueba que la API llega a Redis
#
# El trabajo pesado (el grafo con sus llamadas al LLM) lo hacen los workers (worker.py), nunca el endpoint: por eso el event loop no se bloquea

# Importa AsyncIterator y Callable para tipar la fábrica de runtime
from collections.abc import AsyncIterator, Callable
# Importa los context managers para abrir y cerrar los recursos
from contextlib import AbstractAsyncContextManager, asynccontextmanager
# Importa uuid para generar el job_id
import uuid

# Importa los bloques de FastAPI
from fastapi import FastAPI, HTTPException, Response
# Importa los modelos validados de entrada
from pydantic import BaseModel, Field

# Importa la configuración
from app.config import cargar_configuracion
# Importa el router de aprobación humana
from app.hitl import router as router_hitl
# Importa los modelos de trabajos
from app.jobs import EstadoJob, Job, JobStore
# Importa la observabilidad
from app.observability import cerrar_observabilidad, configurar_logging, iniciar_observabilidad
# Importa el runtime
from app.runtime import Runtime, abrir_runtime
# Importa los workers embebidos
from app.worker import detener_workers, iniciar_workers

# Tipo de la función que abre los recursos del proceso (en los tests se reemplaza por una con Redis falso y LLM falsos)
AbridorRuntime = Callable[[], AbstractAsyncContextManager[Runtime]]


# Cuerpo de POST /tasks
class TareaIn(BaseModel):
    # Pregunta para el orquestador
    pregunta: str = Field(min_length=3, max_length=1000, examples=["¿Qué opinan los clientes sobre la entrega de los auriculares?"])
    # Si se completa, el informe se ENVÍA a esa dirección: es una acción crítica y exige aprobación humana antes de salir
    notificar_a: str = Field(default="", max_length=200, pattern=r"^$|^[^@\s]+@[^@\s]+\.[^@\s]+$", examples=["gerencia@empresa.com"])


# Abre los recursos reales leyendo el entorno (.env): configuración, Phoenix, Redis y, si hay workers embebidos, el grafo con el checkpointer
@asynccontextmanager
async def abrir_desde_entorno() -> AsyncIterator[Runtime]:
    configurar_logging()
    settings = cargar_configuracion()
    proveedor = iniciar_observabilidad(settings)
    try:
        async with abrir_runtime(settings, con_grafo=settings.workers_embebidos > 0) as rt:
            yield rt
    finally:
        cerrar_observabilidad(proveedor)


# Crea la aplicación. "abrir" permite inyectar otros recursos (los tests usan Redis falso y LLM falsos)
def crear_app(abrir: AbridorRuntime = abrir_desde_entorno) -> FastAPI:
    # Arranque y cierre: abre los recursos y levanta los workers embebidos (si se pidieron)
    @asynccontextmanager
    async def lifespan(aplicacion: FastAPI) -> AsyncIterator[None]:
        async with abrir() as rt:
            aplicacion.state.store = rt.store
            tareas = iniciar_workers(rt, rt.settings.workers_embebidos)
            try:
                yield
            finally:
                await detener_workers(tareas)

    aplicacion = FastAPI(
        title="Orquestador multi-agente - API de producción",
        description="Cola de trabajos sobre Redis, workers en segundo plano, aprobación humana (HITL) y trazas en Phoenix.",
        lifespan=lifespan,
        # FastAPI trae su propia telemetría de OpenTelemetry: se apaga el tracing de las peticiones HTTP para que el polling de GET /tasks/{id}
        # no llene Phoenix de trazas diminutas y distorsione el costo y la latencia p95 (las trazas útiles son las del grafo)
        telemetry={"tracing": False},
    )
    aplicacion.include_router(router_hitl)

    # Crea una tarea: guarda el estado PENDING y recién después la encola (así el cliente nunca consulta un id que no existe)
    @aplicacion.post("/tasks", status_code=202, tags=["tareas"])
    async def crear_tarea(cuerpo: TareaIn, respuesta: Response) -> Job:
        store: JobStore = aplicacion.state.store
        job = Job(job_id=str(uuid.uuid4()), status=EstadoJob.PENDING, pregunta=cuerpo.pregunta, notificar_a=cuerpo.notificar_a)
        await store.crear(job)
        await store.encolar(job.job_id)
        respuesta.headers["Location"] = f"/tasks/{job.job_id}"
        return job

    # Consulta el estado de una tarea (el cliente hace polling acá hasta ver DONE o FAILED)
    @aplicacion.get("/tasks/{job_id}", tags=["tareas"])
    async def consultar_tarea(job_id: str) -> Job:
        store: JobStore = aplicacion.state.store
        job = await store.obtener(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Tarea no encontrada")
        return job

    # Salud: confirma que la API llega a Redis
    @aplicacion.get("/health", tags=["sistema"])
    async def salud() -> dict[str, object]:
        store: JobStore = aplicacion.state.store
        try:
            await store.ping()
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Redis no responde: {type(error).__name__}") from error
        return {"status": "ok", "redis": True, "pendientes_en_cola": await store.pendientes()}

    return aplicacion


# Aplicación que levanta uvicorn: uvicorn app.main:app
app = crear_app()
