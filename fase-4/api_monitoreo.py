"""API con registro de predicciones (Fase 4).

Reutiliza la aplicación de la Fase 3 (``crear_app``) y le conecta el
observador de ``/predict`` para que cada intento quede registrado en el
CSV configurado por ``REGISTRO_PREDICCIONES``.

Inicio local, desde la carpeta ``fase-4``::

    uvicorn api_monitoreo:app --host 127.0.0.1 --port 8000

La Fase 3 no se modifica: su gancho ``ObservadorPrediccion`` es el único
punto de integración.
"""

import os
from pathlib import Path

import sys

DIR_FASE4 = Path(__file__).resolve().parent
DIR_FASE3 = DIR_FASE4.parent / "fase-3"

if str(DIR_FASE3) not in sys.path:
    sys.path.insert(0, str(DIR_FASE3))

from api import crear_app  # noqa: E402
from configuracion import Configuracion  # noqa: E402

import registro  # noqa: E402

RUTA_REGISTRO = Path(
    os.environ.get("REGISTRO_PREDICCIONES", "").strip()
    or str(DIR_FASE4 / "logs" / "predicciones.csv")
)


def _observador(estado_http: int, version: str | None, prediccion: bool | None) -> None:
    registro.registrar(RUTA_REGISTRO, estado_http, version, prediccion)


config = Configuracion.desde_entorno()
app = crear_app(config, observador=_observador)
