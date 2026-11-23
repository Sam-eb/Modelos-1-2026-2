"""Registro de solicitudes a /predict en un CSV con modo agregar.

Cada llamada a /predict (exitosa o fallida) genera exactamente una fila con:
fecha_hora, estado_http, version y prediccion.  El encabezado se escribe solo
al crear el archivo.  La carpeta se crea si no existe.
"""

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

ENCABEZADO = ["fecha_hora", "estado_http", "version", "prediccion"]

_bloqueo = threading.Lock()


def registrar(
    ruta: Path,
    estado_http: int,
    version: str | None,
    prediccion: bool | None,
) -> None:
    """Agrega una fila al CSV de registro.  Seguro entre hilos."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with _bloqueo:
        escribir_encabezado = not ruta.is_file() or ruta.stat().st_size == 0
        with ruta.open("a", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            if escribir_encabezado:
                escritor.writerow(ENCABEZADO)
            escritor.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                estado_http,
                version if version is not None else "",
                str(prediccion).lower() if prediccion is not None else "",
            ])
