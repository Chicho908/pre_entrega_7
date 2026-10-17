# Módulo de configuración: lee las variables de entorno y las valida apenas arranca el programa

# Importa os para leer las variables de entorno
import os
# Importa dataclass y field para agrupar toda la configuración en un solo objeto inmutable
from dataclasses import dataclass, field
# Importa Literal, cast y get_args para restringir y validar los proveedores de LLM permitidos
from typing import Literal, cast, get_args

# Importa load_dotenv para cargar el archivo .env cuando se trabaja en local
from dotenv import load_dotenv

# Define los proveedores de LLM que soporta el proyecto
Proveedor = Literal["openai", "anthropic"]

# Define el modelo por defecto de cada proveedor (se puede cambiar con la variable LLM_MODEL)
MODELOS_POR_DEFECTO: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}

# Define el nombre de la variable de entorno donde vive la API key de cada proveedor
VARIABLE_API_KEY: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


# Error propio para avisar de forma clara que una variable de entorno falta o es inválida
class ConfigError(Exception):
    pass


# Agrupa toda la configuración del proyecto en un objeto inmutable (frozen=True)
@dataclass(frozen=True)
class Settings:
    # Proveedor de LLM elegido (openai o anthropic)
    proveedor: Proveedor
    # API key del proveedor (repr=False evita que se imprima por accidente en logs o errores)
    api_key: str = field(repr=False)
    # Nombre del modelo a usar
    modelo: str
    # Tope de pasos del grafo principal por invocación (red de seguridad contra bucles infinitos)
    limite_recursion: int
    # Tope de decisiones del supervisor: al superarlo se fuerza la síntesis con lo que haya
    max_pasos_supervisor: int
    # Veces que un especialista puede repetir su trabajo para corregir un error de validación
    max_refinamientos: int
    # Tope de pasos del ciclo ReAct interno de cada especialista
    limite_iteraciones_agente: int
    # URL de Redis (estado de los trabajos, cola y checkpoints de LangGraph)
    redis_url: str = "redis://localhost:6379"
    # Workers que levanta la propia API como tareas asyncio (0 = los workers corren en otro proceso)
    workers_embebidos: int = 2
    # Trabajos que procesa en paralelo cada proceso worker (python -m app.worker)
    worker_concurrencia: int = 5
    # Tiempo máximo de un trabajo: al superarlo pasa a FAILED en vez de dejar al cliente esperando
    job_timeout_s: int = 300
    # Cuánto viven en Redis el estado de un trabajo y sus checkpoints
    job_ttl_s: int = 86400
    # Endpoint de Phoenix al que se envían las trazas ("" = observabilidad desactivada)
    phoenix_endpoint: str = ""
    # Nombre del proyecto en Phoenix
    phoenix_proyecto: str = "pre-entrega-7"


# Lee una variable de entorno entera y valida que no sea menor al mínimo permitido
def _leer_entero(nombre: str, por_defecto: int, minimo: int) -> int:
    # Lee la variable como texto; si no existe usa el valor por defecto
    crudo = os.getenv(nombre, "").strip() or str(por_defecto)
    # Intenta convertir el texto a entero
    try:
        valor = int(crudo)
    # Si no es un número, avisa con un mensaje claro en vez de dejar un error críptico
    except ValueError as error:
        raise ConfigError(f"{nombre} debe ser un número entero, pero vale '{crudo}'.") from error
    # Valida el mínimo permitido
    if valor < minimo:
        raise ConfigError(f"{nombre} debe ser como mínimo {minimo}, pero vale {valor}.")
    # Devuelve el valor ya validado
    return valor


# Carga y valida toda la configuración; si algo falta, corta con un ConfigError explicativo
def cargar_configuracion() -> Settings:
    # Carga el archivo .env (si existe) a las variables de entorno del proceso
    load_dotenv()

    # Lee el proveedor elegido (por defecto openai) y lo normaliza
    proveedor_crudo = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    # Valida que sea uno de los proveedores soportados
    if proveedor_crudo not in get_args(Proveedor):
        raise ConfigError(
            f"LLM_PROVIDER debe ser uno de {list(get_args(Proveedor))}, pero vale '{proveedor_crudo}'."
        )
    # Le dice al tipado estático que el texto ya fue validado como Proveedor
    proveedor = cast(Proveedor, proveedor_crudo)

    # Busca el nombre de la variable de la API key según el proveedor
    nombre_variable = VARIABLE_API_KEY[proveedor]
    # Lee la API key y le quita espacios
    api_key = os.getenv(nombre_variable, "").strip()
    # Valida que exista y que no sea el valor de ejemplo del .env.example
    if not api_key or api_key.startswith("tu_"):
        raise ConfigError(
            f"Falta {nombre_variable}. Copiá .env.example a .env y completá tu API key."
        )

    # Usa el modelo indicado en LLM_MODEL o, si está vacío, el modelo por defecto del proveedor
    modelo = os.getenv("LLM_MODEL", "").strip() or MODELOS_POR_DEFECTO[proveedor]

    # Lee los topes del grafo para validar que sean coherentes entre sí
    limite_recursion = _leer_entero("RECURSION_LIMIT", por_defecto=40, minimo=4)
    max_pasos_supervisor = _leer_entero("MAX_PASOS_SUPERVISOR", por_defecto=8, minimo=2)
    # Cada decisión del supervisor puede usar hasta 3 pasos del grafo (supervisor, especialista y validación): con menos, el grafo se cortaría antes de poder cerrar
    if limite_recursion < 3 * max_pasos_supervisor:
        raise ConfigError(
            f"RECURSION_LIMIT vale {limite_recursion}, pero con MAX_PASOS_SUPERVISOR={max_pasos_supervisor} necesita ser como mínimo {3 * max_pasos_supervisor}. "
            "Revisá tu .env (si lo copiaste de otra pre-entrega, puede traer un valor más bajo)."
        )

    # Arma y devuelve la configuración completa
    return Settings(
        proveedor=proveedor,
        api_key=api_key,
        modelo=modelo,
        limite_recursion=limite_recursion,
        max_pasos_supervisor=max_pasos_supervisor,
        max_refinamientos=_leer_entero("MAX_REFINAMIENTOS", por_defecto=1, minimo=0),
        limite_iteraciones_agente=_leer_entero("LIMITE_ITERACIONES_AGENTE", por_defecto=8, minimo=2),
        redis_url=os.getenv("REDIS_URL", "").strip() or "redis://localhost:6379",
        workers_embebidos=_leer_entero("WORKERS_EMBEBIDOS", por_defecto=2, minimo=0),
        worker_concurrencia=_leer_entero("WORKER_CONCURRENCIA", por_defecto=5, minimo=1),
        job_timeout_s=_leer_entero("JOB_TIMEOUT_SEGUNDOS", por_defecto=300, minimo=5),
        job_ttl_s=_leer_entero("JOB_TTL_SEGUNDOS", por_defecto=86400, minimo=60),
        phoenix_endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip(),
        phoenix_proyecto=os.getenv("PHOENIX_PROJECT_NAME", "").strip() or "pre-entrega-7",
    )
