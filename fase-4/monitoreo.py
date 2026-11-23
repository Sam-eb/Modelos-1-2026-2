"""Script de monitoreo: lee el registro y muestra indicadores del servicio.

Uso::

    python monitoreo.py
    python monitoreo.py --registro ruta/al/registro.csv
"""

import argparse
import csv
import os
from pathlib import Path

DIR_FASE4 = Path(__file__).resolve().parent
RUTA_POR_DEFECTO = DIR_FASE4 / "logs" / "predicciones.csv"


def leer_registro(ruta: Path) -> list[dict[str, str]]:
    """Lee el CSV de registro y devuelve las filas como diccionarios."""
    if not ruta.is_file():
        return []
    with ruta.open(newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def calcular_indicadores(filas: list[dict[str, str]]) -> tuple[int, float | None]:
    """Devuelve (total, porcentaje_fallidas).  None si no hay solicitudes."""
    total = len(filas)
    if total == 0:
        return 0, None
    fallidas = sum(
        1 for fila in filas if int(fila["estado_http"]) >= 400
    )
    return total, round(fallidas / total * 100, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registro",
        type=Path,
        default=Path(
            os.environ.get("REGISTRO_PREDICCIONES", "").strip()
            or str(RUTA_POR_DEFECTO)
        ),
        help="Ruta al CSV de registro de predicciones.",
    )
    args = parser.parse_args()

    filas = leer_registro(args.registro)
    total, porcentaje = calcular_indicadores(filas)

    print(f"Total de solicitudes: {total}")
    if porcentaje is None:
        print("Solicitudes fallidas: sin solicitudes")
    else:
        print(f"Solicitudes fallidas: {porcentaje}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
