"""Cliente mínimo para probar la API de Titanic Espacial."""

import argparse
import csv
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def construir_pasajero(fila: dict[str, str]) -> dict[str, object]:
    campos_texto = ("HomePlanet", "Cabin")
    campos_numericos = ("Age", "RoomService", "Spa", "VRDeck")
    pasajero: dict[str, object] = {
        "PassengerId": fila.get("PassengerId") or None,
        **{campo: fila.get(campo) or None for campo in campos_texto},
        **{
            campo: float(fila[campo]) if fila.get(campo) else None
            for campo in campos_numericos
        },
    }
    cryosleep = fila.get("CryoSleep", "").strip().lower()
    pasajero["CryoSleep"] = (
        None if not cryosleep else cryosleep == "true"
    )
    return pasajero


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="URL del endpoint /predict.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("..") / "fase-2" / "data" / "ejemplo_prediccion.csv",
        help="CSV de Fase 2; por defecto, su primera fila.",
    )
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as archivo:
        fila = next(csv.DictReader(archivo), None)
    if fila is None:
        parser.error(f"El CSV no contiene filas: {args.csv}")

    cuerpo = json.dumps(construir_pasajero(fila)).encode("utf-8")
    solicitud = Request(
        args.url,
        data=cuerpo,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(solicitud) as respuesta:
            print(json.dumps(json.loads(respuesta.read()), ensure_ascii=False, indent=2))
    except HTTPError as error:
        print(f"HTTP {error.code}")
        print(json.dumps(json.loads(error.read()), ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
