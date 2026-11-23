# Imagen de la API con registro de predicciones (Fase 4).
# Se construye desde la raíz del repositorio, igual que la Fase 3.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/fase-2:/app/fase-3:/app/fase-4"

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app

COPY fase-4/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Código de la Fase 2 (congelado)
COPY fase-2/src/ /app/fase-2/src/
COPY fase-2/train.py /app/fase-2/train.py
COPY fase-2/predict.py /app/fase-2/predict.py

# Código de la Fase 3 (congelado)
COPY fase-3/api.py /app/fase-3/api.py
COPY fase-3/configuracion.py /app/fase-3/configuracion.py
COPY fase-3/errores.py /app/fase-3/errores.py
COPY fase-3/esquemas.py /app/fase-3/esquemas.py
COPY fase-3/servicio_modelo.py /app/fase-3/servicio_modelo.py

# Código de la Fase 4
COPY fase-4/ /app/fase-4/

# Puntos de montaje
RUN mkdir -p /app/fase-2/data /app/fase-2/models /app/fase-4/logs

WORKDIR /app/fase-4
EXPOSE 8000

# El registro se escribe en /app/fase-4/logs (montado desde el host).
CMD ["uvicorn", "api_monitoreo:app", "--host", "0.0.0.0", "--port", "8000"]
