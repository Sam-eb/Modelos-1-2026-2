"""Datos sintéticos y utilidades compartidas por las pruebas.

Las pruebas no usan el dataset real ni la red: generan un CSV pequeño con el
mismo esquema de Kaggle en directorios temporales.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import train

COLUMNAS_KAGGLE = [
    "PassengerId",
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Name",
    "Transported",
]


def crear_datos_sinteticos(n_grupos: int = 50, semilla: int = 0) -> pd.DataFrame:
    """Genera pasajeros con el esquema de Kaggle, agrupados por viaje.

    Cada grupo tiene entre 1 y 3 pasajeros (≈100 filas con 50 grupos).
    `Transported` depende de CryoSleep y del gasto para que el modelo tenga
    señal que aprender.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for grupo in range(1, n_grupos + 1):
        for miembro in range(1, grupo % 3 + 2):
            dormido = bool(rng.random() < 0.4)
            gastos = {
                col: 0.0 if dormido else float(rng.integers(0, 1500))
                for col in ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
            }
            transportado = dormido or (gastos["Spa"] < 300 and rng.random() < 0.6)
            filas.append(
                {
                    "PassengerId": f"{grupo:04d}_{miembro:02d}",
                    "HomePlanet": rng.choice(["Earth", "Europa", "Mars"]),
                    "CryoSleep": dormido,
                    "Cabin": f"{rng.choice(list('ABCDEFG'))}/{grupo}/{rng.choice(['P', 'S'])}",
                    "Destination": rng.choice(["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]),
                    "Age": float(rng.integers(0, 70)),
                    "VIP": bool(rng.random() < 0.05),
                    **gastos,
                    "Name": f"Pasajero {grupo}-{miembro}",
                    "Transported": bool(transportado),
                }
            )
    return pd.DataFrame(filas, columns=COLUMNAS_KAGGLE)


def entrenar_version(
    csv: Path,
    output_dir: Path,
    version: str,
    semilla: int = 0,
) -> int:
    """Ejecuta train.py como desde la línea de comandos (sin CV, por rapidez)."""
    return train.main(
        [
            "--data", str(csv),
            "--target", "Transported",
            "--output-dir", str(output_dir),
            "--version", version,
            "--seed", str(semilla),
            "--cv-folds", "0",
        ]
    )


@pytest.fixture(scope="session")
def datos_entrenamiento() -> pd.DataFrame:
    return crear_datos_sinteticos()


@pytest.fixture(scope="session")
def csv_entrenamiento(tmp_path_factory, datos_entrenamiento) -> Path:
    ruta = tmp_path_factory.mktemp("datos") / "datos.csv"
    datos_entrenamiento.to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="session")
def version_entrenada(tmp_path_factory, csv_entrenamiento) -> Path:
    """Una versión entrenada una sola vez y compartida por varias pruebas."""
    models = tmp_path_factory.mktemp("models")
    assert entrenar_version(csv_entrenamiento, models, "v1") == 0
    return models / "v1"


@pytest.fixture
def datos_prediccion(datos_entrenamiento) -> pd.DataFrame:
    """Pasajeros nuevos (sin la variable objetivo) con columnas en otro orden."""
    nuevos = crear_datos_sinteticos(n_grupos=8, semilla=99).drop(columns="Transported")
    return nuevos[list(reversed(nuevos.columns))]
