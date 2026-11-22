# Python 3.13 coincide con la versión para la que están fijadas las dependencias.
# slim reduce el tamaño sin cambiar la versión del runtime.
FROM python:3.13-slim

# PYTHONPATH permite importar tanto api.py como los módulos src/train/predict
# de Fase 2. La estructura de directorios conserva las rutas usadas por la app.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/fase-2:/app/fase-3"

# La imagen no incluye .git. El build inyecta el commit actual para la
# metadata que registra train.py; el valor por defecto permite builds manuales.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app

# Fase 3 fija también las dependencias compartidas por Fase 2.
COPY fase-3/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Se copia únicamente el código necesario de Fase 2. No se copian datasets ni
# modelos: los datos se montan en /app/fase-2/data (solo lectura) y los modelos
# en /app/fase-2/models (lectura-escritura y persistentes en el host).
COPY fase-2/src/ /app/fase-2/src/
COPY fase-2/train.py /app/fase-2/train.py
COPY fase-2/predict.py /app/fase-2/predict.py
COPY fase-3/ /app/fase-3/

# Las carpetas son puntos de montaje; crearlas no carga ni entrena modelos.
RUN mkdir -p /app/fase-2/data /app/fase-2/models

# Configuracion.desde_entorno admite MODELOS_DIR, MODELO_VERSION,
# DATOS_ENTRENAMIENTO, TARGET, SEMILLA, TEST_SIZE y CV_FOLDS. Sus defaults
# apuntan a /app/fase-2/models y /app/fase-2/data/datos.csv, entre otros.
# Esas rutas se pueden sustituir con `docker run -e NOMBRE=valor`.
WORKDIR /app/fase-3
EXPOSE 8000

# No se entrena durante el build: /train se ejecuta bajo demanda al llamar la API.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
