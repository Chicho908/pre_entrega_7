# Tests del almacén de trabajos y la cola sobre Redis (Redis falso en memoria)

# Importa asyncio para lanzar operaciones concurrentes
import asyncio

# Importa el cliente de Redis
from redis.asyncio import Redis

# Importa el almacén y los modelos
from app.jobs import EstadoJob, Job, JobStore


# Crea un trabajo de ejemplo
def nuevo_job(job_id: str = "job-1") -> Job:
    return Job(job_id=job_id, status=EstadoJob.PENDING, pregunta="¿Qué opinan de las zapatillas?", notificar_a="")


# Test 1: un trabajo guardado se lee igual, con sus marcas de tiempo y un vencimiento (TTL) en Redis
async def test_crear_y_obtener(redis: Redis) -> None:
    store = JobStore(redis, ttl_s=120)
    await store.crear(nuevo_job())
    job = await store.obtener("job-1")
    assert job is not None and job.status == EstadoJob.PENDING and job.pregunta.startswith("¿Qué opinan")
    assert job.resultado is None and job.error is None and job.creado != ""
    assert 0 < await redis.ttl("job:job-1") <= 120


# Test 2: un id desconocido devuelve None (la API lo traduce a 404)
async def test_obtener_inexistente(redis: Redis) -> None:
    assert await JobStore(redis).obtener("no-existe") is None


# Test 3: la transición solo se aplica desde los estados permitidos y guarda campos compuestos
async def test_transicion_con_estado_esperado(redis: Redis) -> None:
    store = JobStore(redis)
    await store.crear(nuevo_job())
    # Desde PENDING no se puede pasar a DONE "saltando" estados que no se declararon
    assert not await store.transicion("job-1", {EstadoJob.RUNNING}, EstadoJob.DONE)
    assert await store.transicion("job-1", {EstadoJob.PENDING}, EstadoJob.RUNNING)
    assert await store.transicion("job-1", {EstadoJob.RUNNING}, EstadoJob.WAITING_APPROVAL, aprobacion_pendiente={"destinatario": "a@b.com"})
    job = await store.obtener("job-1")
    assert job is not None and job.status == EstadoJob.WAITING_APPROVAL and job.aprobacion_pendiente == {"destinatario": "a@b.com"}
    # Vaciar un campo lo deja en None
    assert await store.transicion("job-1", {EstadoJob.WAITING_APPROVAL}, EstadoJob.DONE, resultado="listo", aprobacion_pendiente=None)
    job = await store.obtener("job-1")
    assert job is not None and job.resultado == "listo" and job.aprobacion_pendiente is None


# Test 4: de 20 intentos simultáneos de tomar el mismo trabajo, solo uno lo consigue (no se procesa dos veces)
async def test_transicion_es_atomica(redis: Redis) -> None:
    store = JobStore(redis)
    await store.crear(nuevo_job())
    resultados = await asyncio.gather(*(store.transicion("job-1", {EstadoJob.PENDING}, EstadoJob.RUNNING) for _ in range(20)))
    assert resultados.count(True) == 1


# Test 5: una transición sobre un trabajo que no existe (o que expiró) devuelve False
async def test_transicion_de_inexistente(redis: Redis) -> None:
    assert not await JobStore(redis).transicion("fantasma", {EstadoJob.PENDING}, EstadoJob.RUNNING)


# Test 6: la cola es FIFO, entrega cada id una sola vez y devuelve None si está vacía
async def test_cola_fifo(redis: Redis) -> None:
    store = JobStore(redis)
    await store.encolar("a")
    await store.encolar("b")
    assert await store.pendientes() == 2
    assert await store.siguiente(timeout=1) == "a" and await store.siguiente(timeout=1) == "b"
    assert await store.siguiente(timeout=1) is None
