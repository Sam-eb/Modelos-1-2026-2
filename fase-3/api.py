"""API REST del Titanic Espacial (Fase 3).

Expone el modelo de la Fase 2 mediante tres operaciones:

- GET  /health   estado del servicio.
- POST /predict  predicción de un pasajero con la versión configurada.
- POST /train    entrena y guarda una versión nueva sin cambiar la activa.

Inicio local, desde la carpeta `fase-3`:

    uvicorn api:app --host 127.0.0.1 --port 8000

La configuración se toma de variables de entorno (ver `configuracion.py`), el
contrato de entrada y salida está en `esquemas.py` y los mensajes de error en
`errores.py`.
La lógica de datos, preprocesamiento, modelo y versiones es la de la Fase 2 y
se usa a través de `servicio_modelo.py`.
"""

import threading
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import servicio_modelo
from configuracion import Configuracion
from errores import cuerpo_error_validacion
from esquemas import (
    ETIQUETAS,
    Pasajero,
    RespuestaEntrenamiento,
    RespuestaPrediccion,
    SolicitudEntrenamiento,
)

RUTA_PREDICT = "/predict"

# Observador opcional de cada intento de /predict (lo usa la Fase 4 para el
# registro): recibe el código HTTP, la versión usada (o None) y la predicción
# (o None si hubo error).
ObservadorPrediccion = Callable[[int, str | None, bool | None], None]


def crear_app(
    config: Configuracion | None = None,
    observador: ObservadorPrediccion | None = None,
) -> FastAPI:
    """Crea la aplicación. No carga el modelo: se carga al primer uso."""
    config = config or Configuracion.desde_entorno()
    app = FastAPI(
        title="API Titanic Espacial",
        version="1.0",
        description="Predicción y entrenamiento del modelo de la Fase 2.",
    )
    app.state.config = config
    cache: dict[str, servicio_modelo.ModeloCargado] = {}
    bloqueo_cache = threading.Lock()

    def notificar(estado: int, version: str | None, prediccion: bool | None) -> None:
        if observador is not None:
            observador(estado, version, prediccion)

    def modelo_configurado() -> servicio_modelo.ModeloCargado:
        version = config.modelo_version
        with bloqueo_cache:
            if version not in cache:
                cache[version] = servicio_modelo.cargar_modelo(config.modelos_dir, version)
            return cache[version]

    @app.exception_handler(RequestValidationError)
    async def manejar_validacion(request: Request, error: RequestValidationError):
        if request.url.path == RUTA_PREDICT:
            notificar(status.HTTP_422_UNPROCESSABLE_CONTENT, None, None)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=jsonable_encoder(cuerpo_error_validacion(error)),
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """La API está en ejecución; informa la versión configurada."""
        disponibles = servicio_modelo.versiones_disponibles(config.modelos_dir)
        return {
            "estado": "ok",
            "version_configurada": config.modelo_version,
            "modelo_disponible": config.modelo_version in disponibles,
        }

    @app.post(RUTA_PREDICT, response_model=RespuestaPrediccion)
    def predict(pasajero: Pasajero):
        """Predice si el pasajero fue transportado con la versión configurada."""
        try:
            modelo = modelo_configurado()
        except servicio_modelo.ModeloNoDisponible as error:
            notificar(status.HTTP_503_SERVICE_UNAVAILABLE, None, None)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": f"Modelo no disponible. {error}"},
            )
        try:
            resultado = servicio_modelo.predecir(modelo, [pasajero.a_registro()])
        except ValueError as error:
            # Validación de la Fase 2 (p. ej. CryoSleep o valores no compatibles).
            notificar(status.HTTP_422_UNPROCESSABLE_CONTENT, modelo.version, None)
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": f"Entrada inválida. {error}"},
            )
        fila = resultado.iloc[0]
        prediccion = bool(int(fila["prediccion"]))
        probabilidad = fila.get("probabilidad")
        notificar(status.HTTP_200_OK, modelo.version, prediccion)
        return RespuestaPrediccion(
            PassengerId=pasajero.PassengerId,
            prediccion=prediccion,
            etiqueta=ETIQUETAS[prediccion],
            probabilidad=None if probabilidad is None else float(probabilidad),
            version=modelo.version,
        )

    @app.post(
        "/train",
        response_model=RespuestaEntrenamiento,
        status_code=status.HTTP_201_CREATED,
    )
    def train(solicitud: SolicitudEntrenamiento | None = None):
        """Entrena con la configuración de la Fase 2 y guarda una versión nueva."""
        semilla = config.semilla
        if solicitud is not None and solicitud.semilla is not None:
            semilla = solicitud.semilla
        if not config.datos_entrenamiento.is_file():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "detail": "No se encontró el CSV de entrenamiento configurado "
                    "(variable DATOS_ENTRENAMIENTO)."
                },
            )
        try:
            version, _, metricas = servicio_modelo.entrenar_nueva_version(
                config.datos_entrenamiento,
                config.modelos_dir,
                target=config.target,
                semilla=semilla,
                test_size=config.test_size,
                cv_folds=config.cv_folds,
            )
        except (FileExistsError, OSError, TypeError, ValueError) as error:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"No se pudo entrenar la versión. {error}"},
            )
        return RespuestaEntrenamiento(
            version=version,
            metricas=metricas,
            version_activa=config.modelo_version,
            mensaje=(
                f"Versión '{version}' guardada. /predict sigue usando "
                f"'{config.modelo_version}'; para usar la nueva, reinicie la API "
                f"con MODELO_VERSION={version}."
            ),
        )

    return app


app = crear_app()
