# Módulo de trabajos: el estado de cada tarea vive en Redis (no en la memoria del proceso) y la cola de pendientes también
# Así la API y los workers pueden ser procesos distintos, reiniciarse, y el estado sobrevive
#
# Claves de Redis:
#   job:<id>      hash con el estado del trabajo (status, pregunta, resultado, error...)
#   tasks:queue   lista con los job_id pendientes (RPUSH para encolar, BLPOP para sacar de a uno)

# Importa json para guardar los campos compuestos (diccionarios) dentro del hash
import json
# Importa StrEnum para que los estados sean texto y a la vez un conjunto cerrado
from enum import StrEnum
# Importa datetime para las marcas de tiempo
from datetime import UTC, datetime
# Importa Any para tipar los campos compuestos
from typing import Any

# Importa Request para obtener el almacén desde un endpoint
from fastapi import Request
# Importa el modelo validado del trabajo
from pydantic import BaseModel
# Importa el cliente asíncrono de Redis y el error de transacción optimista
from redis.asyncio import Redis
# Importa el error que lanza Redis cuando otro cliente modificó la clave vigilada
from redis.exceptions import WatchError

# Prefijo de las claves de los trabajos
PREFIJO_JOB = "job:"
# Nombre de la cola de pendientes
COLA = "tasks:queue"


# Estados posibles de un trabajo (son los de la consigna + WAITING_APPROVAL para el flujo HITL)
class EstadoJob(StrEnum):
    # Encolado, todavía no lo tomó ningún worker
    PENDING = "PENDING"
    # Un worker lo está ejecutando
    RUNNING = "RUNNING"
    # Pausado en el nodo de aprobación humana (tarea crítica)
    WAITING_APPROVAL = "WAITING_APPROVAL"
    # Terminó bien
    DONE = "DONE"
    # Falló: el motivo está en el campo error
    FAILED = "FAILED"


# Marca de tiempo actual en ISO 8601 (UTC)
def ahora() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# Un trabajo tal como lo ve el cliente de la API
class Job(BaseModel):
    # Identificador (también es el thread_id de los checkpoints de LangGraph)
    job_id: str
    # Estado actual
    status: EstadoJob
    # Pregunta del usuario
    pregunta: str
    # Destinatario del informe ("" = no se envía nada; si tiene valor la tarea es crítica y exige aprobación)
    notificar_a: str = ""
    # Respuesta final (cuando termina)
    resultado: str | None = None
    # Qué pasó con el envío del informe (solo en tareas críticas)
    despacho: str | None = None
    # Motivo del fallo (cuando status = FAILED)
    error: str | None = None
    # Lo que tiene que revisar el humano (cuando status = WAITING_APPROVAL)
    aprobacion_pendiente: dict[str, Any] | None = None
    # Decisión del revisor
    decision: dict[str, Any] | None = None
    # Marcas de tiempo
    creado: str = ""
    actualizado: str = ""


# Convierte un valor del trabajo en el texto que se guarda en el hash (None → "", diccionario → JSON)
def _a_texto(valor: Any) -> str:
    if valor is None:
        return ""
    if isinstance(valor, dict):
        return json.dumps(valor, ensure_ascii=False)
    return str(valor)


# Convierte un trabajo en el diccionario de strings que se guarda en el hash de Redis
def _a_hash(job: Job) -> dict[str, str]:
    return {nombre: _a_texto(valor) for nombre, valor in job.model_dump(mode="json").items()}


# Convierte el hash de Redis de vuelta en un trabajo
def _desde_hash(datos: dict[str, str]) -> Job:
    crudo: dict[str, Any] = {}
    for nombre, valor in datos.items():
        if nombre in ("aprobacion_pendiente", "decision"):
            crudo[nombre] = json.loads(valor) if valor else None
        elif nombre in ("resultado", "despacho", "error"):
            crudo[nombre] = valor or None
        else:
            crudo[nombre] = valor
    return Job.model_validate(crudo)


# Almacén de trabajos y cola sobre Redis
class JobStore:
    # Guarda el cliente de Redis y cuánto vive cada trabajo
    def __init__(self, redis: Redis, ttl_s: int = 86400) -> None:
        # Cliente asíncrono (decode_responses=True: todo texto)
        self.redis = redis
        # Cuánto vive un trabajo en Redis
        self.ttl_s = ttl_s

    # Clave del hash de un trabajo
    @staticmethod
    def _clave(job_id: str) -> str:
        return f"{PREFIJO_JOB}{job_id}"

    # Comprueba que Redis responde
    async def ping(self) -> bool:
        return bool(await self.redis.ping())

    # Guarda un trabajo nuevo (estado PENDING). Se guarda ANTES de encolar para que el cliente nunca consulte un id que todavía no existe
    async def crear(self, job: Job) -> None:
        job.creado = job.actualizado = ahora()
        clave = self._clave(job.job_id)
        await self.redis.hset(clave, mapping=_a_hash(job))  # type: ignore[arg-type]  # los stubs de redis exigen un Mapping invariante en la clave
        await self.redis.expire(clave, self.ttl_s)

    # Lee un trabajo; devuelve None si no existe (o si expiró)
    async def obtener(self, job_id: str) -> Job | None:
        datos = await self.redis.hgetall(self._clave(job_id))
        # Con decode_responses=True todo viene como texto (los stubs de redis lo tipan como bytes | str)
        return _desde_hash({str(nombre): str(valor) for nombre, valor in datos.items()}) if datos else None

    # Cambia el estado solo si el actual está en "desde" (compare-and-set con WATCH/MULTI). Devuelve False si no se pudo
    # Sirve para que dos workers nunca tomen el mismo trabajo y dos revisores nunca aprueben el mismo trabajo
    async def transicion(self, job_id: str, desde: set[EstadoJob], hacia: EstadoJob, **campos: Any) -> bool:
        clave = self._clave(job_id)
        # Solo se escriben los campos pedidos + el estado + la marca de tiempo
        escribir = {"status": hacia.value, "actualizado": ahora(), **{nombre: _a_texto(valor) for nombre, valor in campos.items()}}
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    # Vigila la clave: si otro cliente la modifica antes del EXEC, la transacción falla y se reintenta
                    await pipe.watch(clave)
                    actual = await pipe.hget(clave, "status")
                    if actual not in {estado.value for estado in desde}:
                        await pipe.unwatch()  # type: ignore[no-untyped-call]
                        return False
                    pipe.multi()  # type: ignore[no-untyped-call]
                    pipe.hset(clave, mapping=escribir)  # type: ignore[arg-type]
                    pipe.expire(clave, self.ttl_s)
                    await pipe.execute()
                    return True
                except WatchError:
                    continue

    # Encola un job_id para que lo tome un worker
    async def encolar(self, job_id: str) -> None:
        await self.redis.rpush(COLA, job_id)

    # Saca el próximo job_id de la cola esperando hasta "timeout" segundos (BLPOP es atómico: dos workers nunca reciben el mismo)
    async def siguiente(self, timeout: int = 1) -> str | None:
        item = await self.redis.blpop([COLA], timeout=timeout)
        return None if item is None else str(item[1])

    # Cantidad de trabajos esperando en la cola
    async def pendientes(self) -> int:
        return int(await self.redis.llen(COLA))


# Dependencia de FastAPI: devuelve el almacén que la aplicación guardó al arrancar
def obtener_store(request: Request) -> JobStore:
    store: JobStore = request.app.state.store
    return store
