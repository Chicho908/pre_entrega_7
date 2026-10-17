# Módulo del worker: saca trabajos de la cola de Redis, ejecuta el grafo en segundo plano y deja el resultado (o el error) en Redis
# Puede correr dentro de la API (WORKERS_EMBEBIDOS) o como proceso aparte: python -m app.worker
#
# Ciclo de vida de un trabajo:
#   PENDING ──(worker lo toma)──→ RUNNING ─┬→ DONE
#                                          ├→ WAITING_APPROVAL ──(POST /approve)──→ PENDING → RUNNING → DONE
#                                          └→ FAILED (cualquier excepción, timeout o apagado del worker)

# Importa asyncio para los workers concurrentes, el timeout y las señales
import asyncio
# Importa logging para dejar registro
import logging
# Importa signal para apagar el worker ordenadamente con Ctrl+C / docker stop
import signal
# Importa sys para el código de salida
import sys
# Importa dataclass para devolver el resultado de una ejecución
from dataclasses import dataclass
# Importa Any y cast para tipar el estado del grafo
from typing import Any, cast

# Importa el tipo de configuración de LangGraph y el comando de reanudación
from langchain_core.runnables import RunnableConfig
# Importa el comando que reanuda un grafo pausado
from langgraph.types import Command

# Importa la configuración y su error propio
from app.config import ConfigError, Settings, cargar_configuracion
# Importa el grafo compilado
from app.graph import GrafoCompilado
# Importa los modelos de trabajos
from app.jobs import EstadoJob, Job
# Importa la observabilidad
from app.observability import cerrar_observabilidad, configurar_logging, iniciar_observabilidad, span_de_job
# Importa el runtime
from app.runtime import Runtime, abrir_runtime
# Importa el estado inicial y el formateo de la traza
from app.state import estado_inicial
# Importa el formateo de los eventos del grafo para el log
from app.traza import formatear_evento

# Crea el logger del módulo
logger = logging.getLogger(__name__)


# Resultado de una ejecución del grafo: terminó o quedó pausado esperando a un humano
@dataclass(frozen=True)
class SalidaGrafo:
    # True si el grafo quedó pausado en la aprobación humana
    pausado: bool
    # Estado del grafo al terminar la ejecución
    valores: dict[str, Any]
    # Lo que tiene que revisar el humano (solo si está pausado)
    pendiente: dict[str, Any] | None


# Ejecuta (o reanuda) el grafo de un trabajo. El thread_id es el job_id: por eso el checkpoint de Redis es "el estado de ese trabajo"
async def ejecutar_grafo(grafo: GrafoCompilado, job: Job, settings: Settings) -> SalidaGrafo:
    config: RunnableConfig = {
        "configurable": {"thread_id": job.job_id},
        "recursion_limit": settings.limite_recursion,
        "run_name": "orquestador",
        "metadata": {"session_id": job.job_id, "job_id": job.job_id},
    }
    # Con decisión humana guardada se REANUDA el grafo pausado; si no, se arranca uno nuevo
    entrada: Any = Command(resume=job.decision) if job.decision is not None else estado_inicial(job.pregunta, job.notificar_a)
    # Recorre el grafo nodo por nodo y deja la traza legible en el log
    async for evento in grafo.astream(entrada, config=config, stream_mode="updates"):
        for nodo, cambios in cast(dict[str, Any], evento).items():
            if nodo == "__interrupt__":
                logger.info("[%s] ⏸ el grafo quedó pausado esperando aprobación humana", job.job_id[:8])
                continue
            for linea in formatear_evento(nodo, cambios):
                logger.info("[%s] %s", job.job_id[:8], linea)
    # Consulta el checkpoint para saber si terminó o quedó pausado
    foto = await grafo.aget_state(config)
    if foto.next:
        interrupciones = [i for tarea in foto.tasks for i in tarea.interrupts]
        pendiente = cast(dict[str, Any], interrupciones[0].value) if interrupciones else {}
        return SalidaGrafo(pausado=True, valores=dict(foto.values), pendiente=pendiente)
    return SalidaGrafo(pausado=False, valores=dict(foto.values), pendiente=None)


# Procesa un trabajo de principio a fin. Nunca deja escapar una excepción (salvo la cancelación del worker): todo error queda en Redis como FAILED
async def procesar_job(rt: Runtime, job_id: str) -> None:
    store = rt.store
    job = await store.obtener(job_id)
    if job is None:
        logger.warning("El trabajo %s no existe (expiró); se descarta", job_id)
        return
    # Toma el trabajo de forma atómica: si otro worker ya lo tomó, o no está pendiente, se descarta
    if not await store.transicion(job_id, {EstadoJob.PENDING}, EstadoJob.RUNNING, error=None):
        logger.warning("El trabajo %s no estaba PENDING; se descarta", job_id)
        return
    if rt.grafo is None:
        await store.transicion(job_id, {EstadoJob.RUNNING}, EstadoJob.FAILED, error="este proceso no tiene el grafo cargado")
        return
    reanudacion = job.decision is not None
    try:
        # El timeout evita que un LLM colgado deje al cliente haciendo polling para siempre
        async with asyncio.timeout(rt.settings.job_timeout_s):
            nombre_span = "job.reanudar" if reanudacion else "job.ejecutar"
            with span_de_job(job_id, nombre_span, job.pregunta, etiquetas=["reanudacion" if reanudacion else "ejecucion"]) as span:
                salida = await ejecutar_grafo(rt.grafo, job, rt.settings)
                span.set_output("pausado: esperando aprobación humana" if salida.pausado else str(salida.valores.get("respuesta_final", "")))
    # Timeout: el trabajo falla con un motivo claro
    except TimeoutError:
        logger.error("[%s] superó el tiempo máximo de %s s", job_id[:8], rt.settings.job_timeout_s)
        await store.transicion(job_id, {EstadoJob.RUNNING}, EstadoJob.FAILED, error=f"tiempo agotado ({rt.settings.job_timeout_s} s)")
    # Apagado del worker: se marca FAILED (con shield para que el guardado no se cancele) y se propaga la cancelación
    except asyncio.CancelledError:
        await asyncio.shield(store.transicion(job_id, {EstadoJob.RUNNING}, EstadoJob.FAILED, error="el worker se detuvo antes de terminar el trabajo"))
        raise
    # Cualquier otra falla (LLM, red, bug): queda registrada y el cliente la ve en GET /tasks/{id}
    except Exception as error:
        logger.exception("[%s] el trabajo falló", job_id[:8])
        await store.transicion(job_id, {EstadoJob.RUNNING}, EstadoJob.FAILED, error=f"{type(error).__name__}: {error}")
    else:
        # Pausado: queda esperando la decisión humana (el checkpoint ya está guardado en Redis)
        if salida.pausado:
            await store.transicion(job_id, {EstadoJob.RUNNING}, EstadoJob.WAITING_APPROVAL, aprobacion_pendiente=salida.pendiente)
            logger.info("[%s] esperando aprobación humana", job_id[:8])
            return
        # Terminado: guarda la respuesta (si el grafo no produjo ninguna, es un fallo)
        respuesta = str(salida.valores.get("respuesta_final", ""))
        if not respuesta:
            await store.transicion(job_id, {EstadoJob.RUNNING}, EstadoJob.FAILED, error="el grafo terminó sin respuesta final")
            return
        despacho = str(salida.valores.get("despacho", "")) or None
        await store.transicion(job_id, {EstadoJob.RUNNING}, EstadoJob.DONE, resultado=respuesta, despacho=despacho, aprobacion_pendiente=None)
        logger.info("[%s] terminado", job_id[:8])


# Bucle de un worker: saca trabajos de la cola de a uno y los procesa. Sobrevive a errores de Redis o de un trabajo puntual
async def bucle_worker(rt: Runtime, numero: int) -> None:
    logger.info("Worker %s iniciado", numero)
    while True:
        try:
            job_id = await rt.store.siguiente(timeout=1)
            if job_id is not None:
                await procesar_job(rt, job_id)
        except asyncio.CancelledError:
            logger.info("Worker %s detenido", numero)
            raise
        except Exception:
            # Por ejemplo, Redis caído un momento: espera y vuelve a intentar sin matar al worker
            logger.exception("Worker %s: error inesperado; reintenta en 1 s", numero)
            await asyncio.sleep(1)


# Arranca N workers concurrentes y devuelve sus tareas (para cancelarlas al cerrar)
def iniciar_workers(rt: Runtime, cantidad: int) -> list["asyncio.Task[None]"]:
    return [asyncio.create_task(bucle_worker(rt, numero)) for numero in range(1, cantidad + 1)]


# Cancela los workers y espera a que terminen
async def detener_workers(tareas: list["asyncio.Task[None]"]) -> None:
    for tarea in tareas:
        tarea.cancel()
    await asyncio.gather(*tareas, return_exceptions=True)


# Proceso worker independiente: python -m app.worker
async def ejecutar_proceso_worker(settings: Settings) -> None:
    proveedor = iniciar_observabilidad(settings)
    try:
        async with abrir_runtime(settings, con_grafo=True) as rt:
            tareas = iniciar_workers(rt, settings.worker_concurrencia)
            logger.info("Worker listo: %s trabajos en paralelo, esperando tareas en Redis", settings.worker_concurrencia)
            # Espera la señal de apagado (docker stop envía SIGTERM)
            parar = asyncio.Event()
            bucle = asyncio.get_running_loop()
            for senal in (signal.SIGINT, signal.SIGTERM):
                try:
                    bucle.add_signal_handler(senal, parar.set)
                except NotImplementedError:
                    pass  # Windows nativo: Ctrl+C llega como KeyboardInterrupt
            await parar.wait()
            logger.info("Apagando workers...")
            await detener_workers(tareas)
    finally:
        cerrar_observabilidad(proveedor)


# Punto de entrada del proceso worker
def main() -> int:
    configurar_logging()
    try:
        settings = cargar_configuracion()
    except ConfigError as error:
        logger.error("Error de configuración: %s", error)
        return 1
    asyncio.run(ejecutar_proceso_worker(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
