# Módulo de Human-in-the-loop (HITL): detecta las tareas críticas, pausa el grafo hasta recibir una aprobación humana y expone el endpoint para aprobar
#
# Qué es "crítico" en este sistema: una tarea que pide ENVIAR el informe a alguien (notificar_a). Es un efecto secundario irreversible:
# una vez enviado, no se puede "desenviar". Por eso el grafo se frena ANTES de enviarlo y espera la decisión de una persona.
#
#   sintetizador ─┬→ (tarea normal) → END
#                 └→ (tarea crítica) → aprobacion ⏸ (interrupt) → despacho (envía solo si fue aprobada) → END

# Importa json para serializar el mensaje del buzón de salida
import json
# Importa logging para dejar registro de lo que se "envía"
import logging
# Importa uuid para generar el identificador de cada envío
import uuid
# Importa Literal y Protocol para tipar las rutas y el despachador
from typing import Annotated, Any, Literal, Protocol

# Importa los bloques de FastAPI para el endpoint de aprobación
from fastapi import APIRouter, Depends, HTTPException
# Importa interrupt: pausa el grafo y devuelve, al reanudar, el valor que mande el humano
from langgraph.types import interrupt
# Importa los modelos validados de la entrada
from pydantic import BaseModel, Field
# Importa el cliente asíncrono de Redis (para el buzón de salida simulado)
from redis.asyncio import Redis

# Importa los modelos y el almacén de trabajos
from app.jobs import EstadoJob, Job, JobStore, obtener_store
# Importa el estado compartido y la pregunta original
from app.agents.synthesis_agent import pregunta_original
# Importa el estado compartido del orquestador
from app.state import EstadoOrquestador

# Crea el logger del módulo
logger = logging.getLogger(__name__)

# Nombre de la lista de Redis donde el despachador simulado deja los mensajes "enviados"
BUZON_SALIDA = "outbox"


# ---------- Despachador: la acción con efectos secundarios ----------


# Contrato del despachador: cualquier cosa que sepa enviar un informe (email, Slack, webhook...)
class Despachador(Protocol):
    # Envía el informe y devuelve el identificador del envío
    async def enviar(self, destinatario: str, asunto: str, cuerpo: str) -> str: ...


# Despachador por defecto: solo deja el envío en el log (no sale nada del sistema)
class DespachadorLog:
    # Envía el informe: solo lo deja en el log y devuelve el id del envío
    async def enviar(self, destinatario: str, asunto: str, cuerpo: str) -> str:
        id_envio = uuid.uuid4().hex[:8]
        logger.info("ENVÍO SIMULADO %s → %s | %s", id_envio, destinatario, asunto)
        return id_envio


# Despachador de la API: simula un email dejando el mensaje en una lista de Redis (se puede ver con redis-cli LRANGE outbox 0 -1)
class DespachadorRedis:
    # Guarda el cliente de Redis donde se deja el buzón de salida
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    # Envía el informe: lo agrega al buzón de salida (outbox) y devuelve el id del envío
    async def enviar(self, destinatario: str, asunto: str, cuerpo: str) -> str:
        id_envio = uuid.uuid4().hex[:8]
        mensaje = json.dumps({"id": id_envio, "para": destinatario, "asunto": asunto, "cuerpo": cuerpo}, ensure_ascii=False)
        await self.redis.rpush(BUZON_SALIDA, mensaje)
        logger.info("ENVÍO SIMULADO %s → %s | %s", id_envio, destinatario, asunto)
        return id_envio


# ---------- Criterio de criticidad ----------


# Devuelve el motivo por el que la tarea es crítica, o "" si no lo es (es el único lugar donde vive la regla)
def motivo_criticidad(estado: EstadoOrquestador) -> str:
    destinatario = estado.get("notificar_a", "").strip()
    if destinatario:
        return f"el sistema va a enviar el informe a '{destinatario}' (efecto secundario irreversible)"
    return ""


# Arista condicional posterior al sintetizador: las tareas críticas pasan por la aprobación humana, las demás terminan
def ruta_tras_sintesis(estado: EstadoOrquestador) -> Literal["aprobacion", "fin"]:
    return "aprobacion" if motivo_criticidad(estado) else "fin"


# ---------- Nodos del grafo ----------


# Decisión que manda el humano al reanudar el grafo
class DecisionHumana(BaseModel):
    # True = enviar el informe; False = cancelar el envío
    aprobado: bool
    # Comentario opcional del revisor (queda en el estado y en la traza)
    comentario: str = Field(default="", max_length=500)


# Nodo de aprobación humana: PAUSA el grafo hasta que alguien decida
# Importante: al reanudar, LangGraph vuelve a ejecutar este nodo desde su primera línea, así que antes del interrupt no puede haber efectos secundarios
async def nodo_aprobacion(state: EstadoOrquestador) -> dict[str, Any]:
    # interrupt guarda el estado en el checkpointer (Redis) y corta la ejecución; lo que se pasa acá es lo que ve el revisor
    respuesta = interrupt(
        {
            "accion": "enviar_informe",
            "motivo": motivo_criticidad(state),
            "destinatario": state["notificar_a"],
            "pregunta": pregunta_original(state),
            "borrador": state["respuesta_final"],
        }
    )
    # Al reanudar, interrupt devuelve lo que se mandó en Command(resume=...): se valida con el contrato
    decision = DecisionHumana.model_validate(respuesta)
    estado = "aprobada" if decision.aprobado else "rechazada"
    return {
        "aprobacion": {"estado": estado, "comentario": decision.comentario},
        "contribuciones": [{"agente": "aprobacion", "intento": 1, "resumen": f"el revisor {estado} el envío", "pasos": [], "error": ""}],
    }


# Crea el nodo de despacho: envía el informe SOLO si el revisor lo aprobó
def crear_nodo_despacho(despachador: Despachador) -> Any:
    # Nodo de despacho: envía el informe solo si el revisor lo aprobó
    async def nodo_despacho(state: EstadoOrquestador) -> dict[str, Any]:
        # Rechazado: no se envía nada
        if state["aprobacion"].get("estado") != "aprobada":
            resultado = "envío cancelado: el revisor rechazó el informe"
        # Aprobado: único punto del sistema con efectos secundarios
        else:
            asunto = f"Informe: {pregunta_original(state)[:80]}"
            id_envio = await despachador.enviar(state["notificar_a"], asunto, state["respuesta_final"])
            resultado = f"informe enviado a {state['notificar_a']} (envío {id_envio})"
        return {
            "despacho": resultado,
            "contribuciones": [{"agente": "despacho", "intento": 1, "resumen": resultado, "pasos": [], "error": ""}],
        }

    return nodo_despacho


# ---------- Endpoint de aprobación ----------

# Router que la API incluye en la aplicación
router = APIRouter(tags=["human-in-the-loop"])


# Cuerpo de POST /tasks/{id}/approve
class AprobacionIn(BaseModel):
    # True = aprobar el envío; False = rechazarlo
    aprobado: bool
    # Comentario opcional
    comentario: str = Field(default="", max_length=500)


# Dependencia: el almacén de trabajos de la aplicación
StoreDep = Annotated[JobStore, Depends(obtener_store)]


# Registra la decisión del revisor y encola la reanudación del grafo (la reanudación la hace un worker, no esta petición)
@router.post("/tasks/{job_id}/approve", status_code=202)
async def aprobar_tarea(job_id: str, cuerpo: AprobacionIn, store: StoreDep) -> Job:
    job = await store.obtener(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    # Transición atómica WAITING_APPROVAL → PENDING: si dos revisores aprueban a la vez, solo uno gana
    decision = DecisionHumana(aprobado=cuerpo.aprobado, comentario=cuerpo.comentario).model_dump()
    reanudada = await store.transicion(job_id, {EstadoJob.WAITING_APPROVAL}, EstadoJob.PENDING, decision=decision)
    if not reanudada:
        raise HTTPException(status_code=409, detail=f"La tarea no está esperando aprobación (estado actual: {job.status.value})")
    # Recién después de guardar la decisión se encola, así el worker siempre la encuentra
    await store.encolar(job_id)
    actualizado = await store.obtener(job_id)
    return actualizado if actualizado is not None else job
