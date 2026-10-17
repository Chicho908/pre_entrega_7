# Prueba de carga: envía N peticiones CONCURRENTES a la API, espera a que terminen y muestra la latencia de cada una (p50 / p95 / máximo)
# Las latencias que ve este script son las del cliente; el costo por ejecución y la latencia p95 "oficiales" se leen en el dashboard de Phoenix
#
# Uso (con la API y los workers levantados):
#   python -m scripts.load_test                       # 5 peticiones contra http://localhost:8000
#   python -m scripts.load_test --n 10 --url http://localhost:8000 --salida resultados.json

# Importa argparse para leer los argumentos
import argparse
# Importa asyncio para lanzar todas las peticiones a la vez
import asyncio
# Importa json para guardar los resultados
import json
# Importa math para el percentil
import math
# Importa Path para guardar el archivo de resultados
from pathlib import Path
# Importa sys para el código de salida
import sys
# Importa time para medir la latencia
import time
# Importa dataclass y asdict para ordenar los resultados
from dataclasses import asdict, dataclass
# Importa Any y cast para tipar las respuestas JSON
from typing import Any, cast

# Importa httpx para llamar a la API
import httpx

# Preguntas de la prueba (todas sobre datos que existen en la base; se reutilizan en orden si N es mayor)
PREGUNTAS: tuple[str, ...] = (
    "¿Qué opinan los clientes sobre la entrega de los auriculares y cómo es el sentimiento general?",
    "¿Cómo es la calidad de las zapatillas y qué rating promedio tienen?",
    "¿Qué dicen los clientes sobre el soporte de la cafetera?",
    "¿Qué opinan del precio del teclado y cómo es el sentimiento?",
    "Comparé la entrega de los auriculares con la de las zapatillas: ¿cuál es mejor valorada?",
)

# Estados en los que el trabajo ya no avanza solo
ESTADOS_FINALES = frozenset({"DONE", "FAILED", "WAITING_APPROVAL"})


# Resultado de una petición
@dataclass(frozen=True)
class Resultado:
    # Número de petición (1..N)
    numero: int
    # Identificador de la tarea
    job_id: str
    # Estado final
    status: str
    # Lo que tardó el POST /tasks en responder (debería ser de milisegundos: no espera al LLM)
    respuesta_post_ms: float
    # Lo que tardó en terminar desde que se envió (latencia completa percibida por el cliente)
    total_s: float
    # Motivo si falló
    error: str | None


# Percentil por el método "nearest-rank" (con 5 datos, el p95 es el mayor)
def percentil(valores: list[float], p: float) -> float:
    ordenados = sorted(valores)
    posicion = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[posicion - 1]


# Envía una petición y hace polling hasta que termina
async def enviar_y_esperar(http: httpx.AsyncClient, numero: int, pregunta: str, timeout_s: float, intervalo_s: float) -> Resultado:
    inicio = time.perf_counter()
    respuesta = await http.post("/tasks", json={"pregunta": pregunta})
    respuesta.raise_for_status()
    post_ms = (time.perf_counter() - inicio) * 1000
    job_id = str(respuesta.json()["job_id"])
    cuerpo: dict[str, Any] = {}
    while time.perf_counter() - inicio < timeout_s:
        cuerpo = cast(dict[str, Any], (await http.get(f"/tasks/{job_id}")).json())
        if cuerpo["status"] in ESTADOS_FINALES:
            break
        await asyncio.sleep(intervalo_s)
    estado = str(cuerpo.get("status", "TIMEOUT"))
    if estado not in ESTADOS_FINALES:
        estado = "TIMEOUT"
    return Resultado(numero, job_id, estado, round(post_ms, 1), round(time.perf_counter() - inicio, 2), cuerpo.get("error"))


# Lanza todas las peticiones A LA VEZ y devuelve los resultados en orden
async def ejecutar_carga(http: httpx.AsyncClient, cantidad: int, timeout_s: float = 300, intervalo_s: float = 1.0) -> list[Resultado]:
    tareas = [enviar_y_esperar(http, n + 1, PREGUNTAS[n % len(PREGUNTAS)], timeout_s, intervalo_s) for n in range(cantidad)]
    return list(await asyncio.gather(*tareas))


# Arma el resumen en texto
def resumen(resultados: list[Resultado]) -> str:
    totales = [r.total_s for r in resultados]
    lineas = ["", f"{'#':>2}  {'job_id':<36}  {'estado':<8}  {'POST (ms)':>9}  {'total (s)':>9}"]
    for r in resultados:
        lineas.append(f"{r.numero:>2}  {r.job_id:<36}  {r.status:<8}  {r.respuesta_post_ms:>9.1f}  {r.total_s:>9.2f}")
        if r.error:
            lineas.append(f"    ERROR: {r.error}")
    lineas += [
        "",
        f"Peticiones: {len(resultados)} | DONE: {sum(r.status == 'DONE' for r in resultados)} | FAILED: {sum(r.status == 'FAILED' for r in resultados)}",
        f"Latencia total (cliente): p50 = {percentil(totales, 50):.2f} s | p95 = {percentil(totales, 95):.2f} s | máx = {max(totales):.2f} s",
        f"El POST /tasks respondió en {max(r.respuesta_post_ms for r in resultados):.0f} ms como máximo: la API no espera al LLM.",
        "Costo por ejecución y latencia p95 de la plataforma: abrí Phoenix (http://localhost:6006).",
    ]
    return "\n".join(lineas)


# Punto de entrada
async def principal() -> int:
    parser = argparse.ArgumentParser(description="Prueba de carga: N peticiones concurrentes contra la API")
    parser.add_argument("--url", default="http://localhost:8000", help="URL base de la API")
    parser.add_argument("--n", type=int, default=5, help="Cantidad de peticiones concurrentes (la consigna pide 5)")
    parser.add_argument("--timeout", type=float, default=300, help="Segundos máximos de espera por petición")
    parser.add_argument("--salida", help="Archivo JSON donde guardar los resultados")
    argumentos = parser.parse_args()
    async with httpx.AsyncClient(base_url=argumentos.url, timeout=30) as http:
        try:
            (await http.get("/health")).raise_for_status()
        except httpx.HTTPError as error:
            print(f"No se pudo conectar con la API en {argumentos.url}: {error}")
            return 1
        print(f"Enviando {argumentos.n} peticiones concurrentes a {argumentos.url} ...")
        resultados = await ejecutar_carga(http, argumentos.n, argumentos.timeout)
    print(resumen(resultados))
    if argumentos.salida:
        contenido = json.dumps([asdict(r) for r in resultados], ensure_ascii=False, indent=2)
        await asyncio.to_thread(Path(argumentos.salida).write_text, contenido, "utf-8")
    return 0 if all(r.status == "DONE" for r in resultados) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(principal()))
