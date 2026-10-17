# Configuración común de los tests: un Redis falso en memoria para cada test

# Importa AsyncIterator para tipar la fixture
from collections.abc import AsyncIterator

# Importa fakeredis: un Redis en memoria que habla el mismo protocolo (los tests no necesitan Docker)
import fakeredis
# Importa pytest_asyncio para las fixtures asíncronas
import pytest_asyncio
# Importa el cliente real de Redis (solo para tipar)
from redis.asyncio import Redis


# Cliente de Redis para los tests: un Redis falso, vacío al empezar y al terminar cada test
@pytest_asyncio.fixture
async def redis() -> AsyncIterator[Redis]:
    cliente: Redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    await cliente.flushdb()
    yield cliente
    await cliente.flushdb()
    await cliente.aclose()
