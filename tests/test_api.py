# Tests de la API: contratos HTTP, que NO bloquea, y los flujos completos (con worker embebido, LLM falsos y Redis falso)

# Importa asyncio para esperar el polling
import asyncio
# Importa AsyncIterator para tipar los recursos de prueba
from collections.abc import AsyncIterator
# Importa los context managers para armar los recursos de prueba
from contextlib import AbstractAsyncContextManager, asynccontextmanager
# Importa cast y Any
from typing import Any, cast

# Importa httpx para llamar a la API en memoria
import httpx
# Importa el checkpointer en memoria
from langgraph.checkpoint.memory import InMemorySaver
# Importa el cliente de Redis
from redis.asyncio import Redis

# Importa el almacén de trabajos
from app.jobs import JobStore
# Importa la aplicación y el tipo de la función que abre los recursos
from app.main import AbridorRuntime, crear_app
# Importa los recursos que comparten la API y el worker
from app.runtime import Runtime
# Importa los dobles de prueba
from tests.fakes import (
    DespachadorEspia, GrafoFalso, armar_orquestador, guion_analista, guion_investigador, guion_supervisor_feliz, guion_sintetizador,
    settings_de_prueba,
)

# Pregunta de los tests
PREGUNTA = "¿Qué opinan los clientes sobre la entrega de los auriculares y cómo es el sentimiento general?"


# Arma la función que abre los recursos con el grafo real y LLM falsos; devuelve también el despachador espía
def abridor_real(redis: Redis, workers: int = 1) -> tuple[AbridorRuntime, DespachadorEspia]:
    espia = DespachadorEspia()

    # Abre los recursos de la aplicación con el grafo real y el despachador espía
    @asynccontextmanager
    async def abrir() -> AsyncIterator[Runtime]:
        grafo, _ = armar_orquestador(
            guion_supervisor_feliz(), guion_investigador(), guion_analista(), guion_sintetizador(), checkpointer=InMemorySaver(), despachador=espia
        )
        yield Runtime(settings_de_prueba(workers_embebidos=workers), JobStore(redis), grafo)

    return abrir, espia


# Arma la función que abre los recursos con un grafo falso
def abridor_falso(redis: Redis, grafo: GrafoFalso, workers: int = 1) -> AbridorRuntime:
    # Abre los recursos de la aplicación con el grafo falso
    @asynccontextmanager
    async def abrir() -> AsyncIterator[Runtime]:
        yield Runtime(settings_de_prueba(workers_embebidos=workers), JobStore(redis), cast(Any, grafo))

    return abrir


# Levanta la app (con su lifespan) y entrega un cliente HTTP en memoria
@asynccontextmanager
async def cliente(abrir: AbridorRuntime) -> AsyncIterator[httpx.AsyncClient]:
    aplicacion = crear_app(abrir)
    lifespan: AbstractAsyncContextManager[Any] = aplicacion.router.lifespan_context(aplicacion)
    async with lifespan:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=aplicacion), base_url="http://prueba") as http:
            yield http


# Hace polling del trabajo hasta que llegue a uno de los estados esperados (falla si no llega a tiempo)
async def esperar(http: httpx.AsyncClient, job_id: str, estados: set[str], intentos: int = 100) -> dict[str, Any]:
    for _ in range(intentos):
        cuerpo = cast(dict[str, Any], (await http.get(f"/tasks/{job_id}")).json())
        if cuerpo["status"] in estados:
            return cuerpo
        await asyncio.sleep(0.05)
    raise AssertionError(f"el trabajo {job_id} no llegó a {estados}")


# Test 1: POST /tasks responde 202 con un job_id y el trabajo en PENDING, sin esperar al grafo (sin workers queda encolado)
async def test_post_encola_y_responde_sin_esperar(redis: Redis) -> None:
    async with cliente(abridor_falso(redis, GrafoFalso(demora_s=30), workers=0)) as http:
        respuesta = await http.post("/tasks", json={"pregunta": PREGUNTA})
        assert respuesta.status_code == 202
        cuerpo = respuesta.json()
        assert cuerpo["status"] == "PENDING" and respuesta.headers["location"] == f"/tasks/{cuerpo['job_id']}"
        # El estado ya está en Redis y el id está en la cola
        assert (await http.get(f"/tasks/{cuerpo['job_id']}")).json()["status"] == "PENDING"
        assert (await http.get("/health")).json() == {"status": "ok", "redis": True, "pendientes_en_cola": 1}


# Test 2: las entradas inválidas se rechazan con 422 y un id desconocido da 404
async def test_validaciones_y_404(redis: Redis) -> None:
    async with cliente(abridor_falso(redis, GrafoFalso(), workers=0)) as http:
        assert (await http.post("/tasks", json={"pregunta": "a"})).status_code == 422
        assert (await http.post("/tasks", json={"pregunta": PREGUNTA, "notificar_a": "no-es-un-mail"})).status_code == 422
        assert (await http.post("/tasks", json={})).status_code == 422
        assert (await http.get("/tasks/no-existe")).status_code == 404
        assert (await http.post("/tasks/no-existe/approve", json={"aprobado": True})).status_code == 404


# Test 3: flujo completo: POST → polling → DONE con la respuesta
async def test_flujo_completo_hasta_done(redis: Redis) -> None:
    abrir, espia = abridor_real(redis)
    async with cliente(abrir) as http:
        job_id = (await http.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]
        final = await esperar(http, job_id, {"DONE", "FAILED"})
        assert final["status"] == "DONE" and "2 reseñas" in final["resultado"] and final["error"] is None
        assert espia.enviados == []


# Test 4: si el grafo falla en segundo plano, GET /tasks/{id} devuelve FAILED con el motivo (no hay polling infinito)
async def test_falla_en_segundo_plano_queda_en_failed(redis: Redis) -> None:
    async with cliente(abridor_falso(redis, GrafoFalso(error=ValueError("boom")))) as http:
        job_id = (await http.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]
        final = await esperar(http, job_id, {"DONE", "FAILED"})
        assert final["status"] == "FAILED" and "ValueError: boom" in final["error"]


# Test 5: flujo HITL completo por la API: pausa → aprobación → reanudación → envío
async def test_flujo_hitl_aprobado(redis: Redis) -> None:
    abrir, espia = abridor_real(redis)
    async with cliente(abrir) as http:
        job_id = (await http.post("/tasks", json={"pregunta": PREGUNTA, "notificar_a": "gerencia@empresa.com"})).json()["job_id"]
        pausado = await esperar(http, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
        assert pausado["status"] == "WAITING_APPROVAL" and espia.enviados == []
        assert pausado["aprobacion_pendiente"]["destinatario"] == "gerencia@empresa.com" and pausado["aprobacion_pendiente"]["borrador"]
        aprobado = await http.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "ok"})
        assert aprobado.status_code == 202
        final = await esperar(http, job_id, {"DONE", "FAILED"})
        assert final["status"] == "DONE" and "informe enviado a gerencia@empresa.com" in final["despacho"]
        assert len(espia.enviados) == 1
        # Aprobar de nuevo es un conflicto: la tarea ya no espera aprobación
        assert (await http.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409


# Test 6: si el revisor rechaza, la tarea termina en DONE pero NO se envía nada
async def test_flujo_hitl_rechazado(redis: Redis) -> None:
    abrir, espia = abridor_real(redis)
    async with cliente(abrir) as http:
        job_id = (await http.post("/tasks", json={"pregunta": PREGUNTA, "notificar_a": "gerencia@empresa.com"})).json()["job_id"]
        await esperar(http, job_id, {"WAITING_APPROVAL"})
        assert (await http.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "no"})).status_code == 202
        final = await esperar(http, job_id, {"DONE", "FAILED"})
        assert final["status"] == "DONE" and "cancelado" in final["despacho"] and espia.enviados == []


# Test 7: no se puede aprobar una tarea que no está esperando aprobación (por ejemplo, una que sigue PENDING)
async def test_aprobar_tarea_no_pausada_da_409(redis: Redis) -> None:
    async with cliente(abridor_falso(redis, GrafoFalso(), workers=0)) as http:
        job_id = (await http.post("/tasks", json={"pregunta": PREGUNTA})).json()["job_id"]
        respuesta = await http.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
        assert respuesta.status_code == 409 and "PENDING" in respuesta.json()["detail"]


# Test 8: dos aprobaciones simultáneas: una gana (202) y la otra recibe 409; el informe se envía una sola vez
async def test_doble_aprobacion_simultanea(redis: Redis) -> None:
    abrir, espia = abridor_real(redis)
    async with cliente(abrir) as http:
        job_id = (await http.post("/tasks", json={"pregunta": PREGUNTA, "notificar_a": "gerencia@empresa.com"})).json()["job_id"]
        await esperar(http, job_id, {"WAITING_APPROVAL"})
        respuestas = await asyncio.gather(*(http.post(f"/tasks/{job_id}/approve", json={"aprobado": True}) for _ in range(5)))
        assert sorted(r.status_code for r in respuestas) == [202, 409, 409, 409, 409]
        await esperar(http, job_id, {"DONE", "FAILED"})
        assert len(espia.enviados) == 1


# Test 9: 5 peticiones concurrentes (la prueba de carga de la consigna) terminan todas, sin bloquear la API
async def test_cinco_peticiones_concurrentes(redis: Redis) -> None:
    grafo = GrafoFalso(demora_s=0.3)
    async with cliente(abridor_falso(redis, grafo, workers=5)) as http:
        respuestas = await asyncio.gather(*(http.post("/tasks", json={"pregunta": PREGUNTA}) for _ in range(5)))
        assert all(r.status_code == 202 for r in respuestas)
        ids = {r.json()["job_id"] for r in respuestas}
        assert len(ids) == 5
        finales = await asyncio.gather(*(esperar(http, i, {"DONE", "FAILED"}) for i in ids))
        assert all(f["status"] == "DONE" for f in finales)
        assert grafo.maximo_en_paralelo == 5


# Test 10: el script de la prueba de carga corre de punta a punta contra la API y calcula bien los percentiles
async def test_script_de_prueba_de_carga(redis: Redis) -> None:
    # Se importa acá porque solo este test usa el script
    from scripts.load_test import ejecutar_carga, percentil, resumen

    assert percentil([1, 2, 3, 4, 5], 95) == 5 and percentil([1, 2, 3, 4, 5], 50) == 3 and percentil([7.0], 95) == 7.0
    async with cliente(abridor_falso(redis, GrafoFalso(demora_s=0.2), workers=5)) as http:
        resultados = await ejecutar_carga(http, 5, timeout_s=20, intervalo_s=0.05)
    assert len(resultados) == 5 and all(r.status == "DONE" for r in resultados)
    assert len({r.job_id for r in resultados}) == 5
    texto = resumen(resultados)
    assert "p95" in texto and "DONE: 5" in texto
