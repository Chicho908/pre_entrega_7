# Imagen de la API y del worker (el mismo codigo; cambia el comando de arranque)
FROM python:3.12-slim

# Logs sin buffer (se ven al instante en docker compose logs) y sin archivos .pyc
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Primero las dependencias: si solo cambia el codigo, Docker reutiliza esta capa
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Despues el codigo
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

# Por defecto arranca la API; el servicio "worker" del compose lo reemplaza por: python -m app.worker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
