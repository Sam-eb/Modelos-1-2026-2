"""Generación de predicciones con una versión guardada del modelo."""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.almacenamiento import cargar_version
from src.datos import cargar_datos_prediccion
from src.preprocesamiento import FEATURES, preparar_datos


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera predicciones del Titanic Espacial con un modelo guardado."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Carpeta de la versión (p. ej. models/v1) o su archivo modelo.joblib.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al CSV con los pasajeros que se van a predecir.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta del CSV donde se guardarán las predicciones.",
    )
    return parser


def _generar_predicciones(
    modelo: Any,
    datos: pd.DataFrame,
) -> pd.DataFrame:
    # preparar_datos selecciona FEATURES, por lo que las columnas adicionales se ignoran.
    features = preparar_datos(datos)
    predicciones = modelo.predict(features)

    resultado = pd.DataFrame({"prediccion": predicciones}, index=datos.index)
    if "PassengerId" in datos.columns:
        resultado.insert(0, "PassengerId", datos["PassengerId"])

    predict_proba = getattr(modelo, "predict_proba", None)
    if callable(predict_proba):
        probabilidades = np.asarray(predict_proba(features))
        clases = getattr(modelo, "classes_", None)
        if probabilidades.ndim != 2 or clases is None:
            raise ValueError("El modelo devolvió probabilidades con un formato inválido.")
        posiciones_clase_1 = np.flatnonzero(np.asarray(clases) == 1)
        if len(posiciones_clase_1) == 1:
            resultado["probabilidad"] = probabilidades[:, posiciones_clase_1[0]]

    # Las categorías no vistas se gestionan con handle_unknown="ignore" en el pipeline.
    if len(resultado) != len(datos):
        raise ValueError("El número de predicciones no coincide con las filas de entrada.")
    return resultado.reset_index(drop=True)


def _guardar_csv_atomico(resultado: pd.DataFrame, ruta_salida: Path) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_temporal: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{ruta_salida.name}.",
            dir=ruta_salida.parent,
            delete=False,
        ) as archivo:
            ruta_temporal = Path(archivo.name)
            resultado.to_csv(archivo, index=False)
        os.replace(ruta_temporal, ruta_salida)
    except Exception:
        if ruta_temporal is not None:
            ruta_temporal.unlink(missing_ok=True)
        raise


def _predecir(args: argparse.Namespace) -> tuple[int, Path]:
    ruta_modelo = Path(args.model)
    ruta_entrada = Path(args.input)
    ruta_salida = Path(args.output)

    # Se acepta la carpeta de la versión o su archivo modelo.joblib.
    if ruta_modelo.is_file() and ruta_modelo.name == "modelo.joblib":
        ruta_modelo = ruta_modelo.parent
    if not ruta_modelo.is_dir():
        raise FileNotFoundError(
            "El modelo no existe: indique la carpeta de la versión "
            f"(p. ej. models/v1) o su archivo modelo.joblib. Ruta recibida: {ruta_modelo}"
        )
    if not ruta_entrada.is_file():
        raise FileNotFoundError(
            f"El archivo de entrada no existe o no es un archivo: {ruta_entrada}"
        )
    if ruta_salida.resolve() == ruta_entrada.resolve():
        raise ValueError("La ruta de salida no puede ser la misma que la del CSV de entrada.")

    modelo, _, metadata = cargar_version(ruta_modelo)
    features_version = metadata.get("features")
    if features_version != list(FEATURES):
        raise ValueError(
            f"La versión '{ruta_modelo}' espera las variables {features_version}, "
            f"pero este código usa {list(FEATURES)}; no son compatibles."
        )
    datos = cargar_datos_prediccion(ruta_entrada)
    resultado = _generar_predicciones(modelo, datos)
    _guardar_csv_atomico(resultado, ruta_salida)
    return len(resultado), ruta_salida


def main(argv: Sequence[str] | None = None) -> int:
    parser = crear_parser()
    args = parser.parse_args(argv)
    try:
        cantidad, ruta_salida = _predecir(args)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))

    print(f"Predicciones generadas: {cantidad}")
    print(f"Archivo guardado en: {ruta_salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
