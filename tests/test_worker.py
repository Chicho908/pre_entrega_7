# Tests del worker: ejecuta el grafo en segundo plano y deja SIEMPRE un estado final en Redis (DONE, FAILED o WAITING_APPROVAL)

# Importa asyncio para esperar y lanzar workers
import asyncio
# Importa cast y Any para pasar grafos falsos
from typing import Any, cast

# Importa el checkpointer en memoria
from langgraph.checkpoint.memory import InMemorySaver
# Importa el cliente de Redis
from redis.asyncio import Redis

# Importa los modelos y el almacén de trabajos
from app.jobs import EstadoJob, Job, JobStore
# Importa los recursos del proceso
from app.runtime import Runtime
# Importa las funciones del worker a probar
from app.worker import bucle_worker, procesar_job
# Importa los dobles de prueba
from tests.fakes import (
    DespachadorEspia, GrafoFalso, armar_orquestador, guion_analista, guion_investigador, guion_supervisor_feliz, guion_sintetizador,
    settings_de_prueba,
)

# Pregunta de los tests
PREGUNTA = "¿Qué opinan los clientes sobre la entrega de los auriculares y cómo es el sentimiento general?"


# Arma un runtime con el grafo REAL y LLM falsos (guiones de la PE6)
def runtime_real(store: JobStore, memoria: InMemorySaver | None = None, espia: DespachadorEspia | None = None, **ajustes: Any) -> Runtime:
    grafo, _ = armar_orquestador(
        guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador(),
        checkpointer=memoria or InMemorySaver(), despachador=espia or DespachadorEspia(),
    )
    return Runtime(settings_de_prueba(**ajustes), store, grafo)


# Arma un runtime con un grafo falso (para forzar demoras y fallas)
def runtime_falso(store: JobStore, grafo: GrafoFalso, **ajustes: Any) -> Runtime:
    return Runtime(settings_de_prueba(**ajustes), store, cast(Any, grafo))


# Crea el trabajo en Redis y lo encola, como hace la API
async def enviar(store: JobStore, job_id: str, notificar_a: str = "") -> None:
    await store.crear(Job(job_id=job_id, status=EstadoJob.PENDING, pregunta=PREGUNTA, notificar_a=notificar_a))
    await store.encolar(job_id)


# Lee un trabajo que sabemos que existe
async def leer(store: JobStore, job_id: str) -> Job:
    job = await store.obtener(job_id)
    assert job is not None
    return job


# Devuelve el id obtenido de la cola (falla el test si estaba vacía)
def id_de_cola(job_id: str | None) -> str:
    assert job_id is not None
    return job_id


# Test 1: camino feliz: PENDING → DONE con la respuesta del grafo guardada en Redis
async def test_trabajo_termina_en_done(redis: Redis) -> None:
    store = JobStore(redis)
    await enviar(store, "j1")
    await procesar_job(runtime_real(store), "j1")
    job = await leer(store, "j1")
    assert job.status == EstadoJob.DONE and job.resultado is not None and "2 reseñas" in job.resultado
    assert job.error is None and job.despacho is None


# Test 2: si el grafo lanza una excepción, el trabajo pasa a FAILED con el motivo (el cliente no queda esperando para siempre)
async def test_excepcion_del_grafo_deja_el_trabajo_en_failed(redis: Redis) -> None:
    store = JobStore(redis)
    await enviar(store, "j2")
    await procesar_job(runtime_falso(store, GrafoFalso(error=RuntimeError("se cayó el proveedor del LLM"))), "j2")
    job = await leer(store, "j2")
    assert job.status == EstadoJob.FAILED and job.error is not None
    assert "RuntimeError" in job.error and "se cayó el proveedor del LLM" in job.error
    assert job.resultado is None


# Test 3: una falla REAL del grafo (se supera el límite de recursión) también termina en FAILED
async def test_limite_de_recursion_termina_en_failed(redis: Redis) -> None:
    store = JobStore(redis)
    await enviar(store, "j3")
    await procesar_job(runtime_real(store, limite_recursion=3), "j3")
    job = await leer(store, "j3")
    assert job.status == EstadoJob.FAILED and job.error is not None and "GraphRecursionError" in job.error


# Test 4: un grafo colgado se corta por timeout y queda FAILED con el motivo
async def test_timeout_termina_en_failed(redis: Redis) -> None:
    store = JobStore(redis)
    await enviar(store, "j4")
    await procesar_job(runtime_falso(store, GrafoFalso(demora_s=30), job_timeout_s=1), "j4")
    job = await leer(store, "j4")
    assert job.status == EstadoJob.FAILED and job.error is not None and "tiempo agotado" in job.error


# Test 5: si el worker se apaga a mitad de un trabajo, queda FAILED (no RUNNING para siempre) y la cancelación se propaga
async def test_apagado_del_worker_deja_el_trabajo_en_failed(redis: Redis) -> None:
    store = JobStore(redis)
    await enviar(store, "j5")
    tarea = asyncio.create_task(procesar_job(runtime_falso(store, GrafoFalso(demora_s=30)), "j5"))
    await asyncio.sleep(0.2)
    tarea.cancel()
    try:
        await tarea
    except asyncio.CancelledError:
        pass
    job = await leer(store, "j5")
    assert job.status == EstadoJob.FAILED and job.error is not None and "worker se detuvo" in job.error


# Test 6: una tarea crítica queda en WAITING_APPROVAL; con la decisión se reanuda y termina en DONE enviando el informe
async def test_tarea_critica_espera_aprobacion_y_se_reanuda(redis: Redis) -> None:
    store = JobStore(redis)
    espia = DespachadorEspia()
    rt = runtime_real(store, espia=espia)
    await enviar(store, "j6", notificar_a="gerencia@empresa.com")
    await procesar_job(rt, "j6")
    job = await leer(store, "j6")
    assert job.status == EstadoJob.WAITING_APPROVAL and job.resultado is None and espia.enviados == []
    assert job.aprobacion_pendiente is not None and job.aprobacion_pendiente["destinatario"] == "gerencia@empresa.com"
    # La API guarda la decisión (WAITING_APPROVAL → PENDING) y vuelve a encolar el trabajo
    assert await store.transicion("j6", {EstadoJob.WAITING_APPROVAL}, EstadoJob.PENDING, decision={"aprobado": True, "comentario": "ok"})
    await store.encolar("j6")
    await procesar_job(rt, id_de_cola(await store.siguiente()))
    job = await leer(store, "j6")
    assert job.status == EstadoJob.DONE and job.resultado is not None and job.aprobacion_pendiente is None
    assert job.despacho is not None and "informe enviado a gerencia@empresa.com" in job.despacho
    assert len(espia.enviados) == 1


# Test 7: la reanudación la puede hacer OTRO worker (otro grafo con el mismo checkpointer): el estado no vive en la memoria del proceso
async def test_reanudacion_desde_otro_worker(redis: Redis) -> None:
    store = JobStore(redis)
    memoria = InMemorySaver()
    espia = DespachadorEspia()
    await enviar(store, "j7", notificar_a="gerencia@empresa.com")
    await procesar_job(runtime_real(store, memoria, espia), "j7")
    # Otro proceso: grafo nuevo, LLM nuevos sin guion (no los necesita), mismo almacén de checkpoints y de trabajos
    grafo_nuevo, _ = armar_orquestador([], [], [], [], checkpointer=memoria, despachador=espia)
    worker_b = Runtime(settings_de_prueba(), JobStore(redis), grafo_nuevo)
    assert await store.transicion("j7", {EstadoJob.WAITING_APPROVAL}, EstadoJob.PENDING, decision={"aprobado": False, "comentario": "no"})
    await procesar_job(worker_b, "j7")
    job = await leer(store, "j7")
    assert job.status == EstadoJob.DONE and job.despacho is not None and "cancelado" in job.despacho and espia.enviados == []


# Test 8: si el mismo id está dos veces en la cola, el segundo intento se descarta (ya no está PENDING) y el grafo corre una sola vez
async def test_trabajo_duplicado_se_procesa_una_vez(redis: Redis) -> None:
    store = JobStore(redis)
    grafo = GrafoFalso()
    rt = runtime_falso(store, grafo)
    await enviar(store, "j8")
    await asyncio.gather(procesar_job(rt, "j8"), procesar_job(rt, "j8"))
    assert grafo.ejecuciones == 1 and (await leer(store, "j8")).status == EstadoJob.DONE


# Test 9: un id inexistente no rompe al worker
async def test_trabajo_inexistente_se_ignora(redis: Redis) -> None:
    await procesar_job(runtime_falso(JobStore(redis), GrafoFalso()), "fantasma")


# Test 10: varios workers procesan trabajos EN PARALELO (3 trabajos de 0,5 s terminan en ~0,5 s, no en 1,5 s)
async def test_workers_en_paralelo(redis: Redis) -> None:
    store = JobStore(redis)
    grafo = GrafoFalso(demora_s=0.5)
    rt = runtime_falso(store, grafo)
    for n in range(3):
        await enviar(store, f"p{n}")
    tareas = [asyncio.create_task(bucle_worker(rt, n)) for n in range(3)]
    try:
        for _ in range(60):
            estados = [(await leer(store, f"p{n}")).status for n in range(3)]
            if all(e == EstadoJob.DONE for e in estados):
                break
            await asyncio.sleep(0.05)
        assert all(e == EstadoJob.DONE for e in estados)
        assert grafo.maximo_en_paralelo == 3
    finally:
        for tarea in tareas:
            tarea.cancel()
        await asyncio.gather(*tareas, return_exceptions=True)


# Test 11: si Redis falla un momento, el worker NO muere: se recupera y sigue procesando
async def test_el_worker_sobrevive_a_un_error_inesperado(redis: Redis) -> None:
    store = JobStore(redis)
    rt = runtime_falso(store, GrafoFalso())
    llamadas = {"n": 0}
    original = store.siguiente

    # Lectura de la cola que falla la primera vez (como un Redis caído un instante) y después funciona
    async def siguiente_con_falla(timeout: int = 1) -> str | None:
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            raise ConnectionError("Redis no responde")
        return await original(timeout)

    store.siguiente = siguiente_con_falla  # type: ignore[method-assign]
    await enviar(store, "r1")
    tarea = asyncio.create_task(bucle_worker(rt, 1))
    try:
        for _ in range(60):
            if (await leer(store, "r1")).status == EstadoJob.DONE:
                break
            await asyncio.sleep(0.1)
        assert (await leer(store, "r1")).status == EstadoJob.DONE
    finally:
        tarea.cancel()
        await asyncio.gather(tarea, return_exceptions=True)


# Error que imita al de un cliente de LLM cuando la API key es inválida (los clientes reales traen el código HTTP en status_code)
class ErrorCredenciales(Exception):
    status_code = 401


# Test 12: con la API key inválida el trabajo termina en FAILED (no en DONE con una respuesta de error disfrazada)
async def test_api_key_invalida_termina_en_failed(redis: Redis) -> None:
    store = JobStore(redis)
    grafo, _ = armar_orquestador(
        guion_supervisor_feliz(), [ErrorCredenciales("Incorrect API key provided")], guion_analista(), guion_sintetizador(),
        checkpointer=InMemorySaver(), despachador=DespachadorEspia(),
    )
    await enviar(store, "j12")
    await procesar_job(Runtime(settings_de_prueba(), store, grafo), "j12")
    job = await leer(store, "j12")
    assert job.status == EstadoJob.FAILED and job.error is not None and "ErrorCredenciales" in job.error
