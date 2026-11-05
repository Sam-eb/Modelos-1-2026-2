"""Partición entrenamiento/prueba y validación cruzada por grupos de viaje.

Se comparte entre `train.py` y `seleccion_variables.py` para que ambos usen
exactamente las mismas particiones con la misma semilla.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def dividir_por_grupos(
    df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    semilla: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve los índices posicionales de entrenamiento y prueba.

    Todos los pasajeros de un mismo grupo quedan del mismo lado.
    """
    if not 0 < test_size < 1:
        raise ValueError("--test-size debe ser un valor mayor que 0 y menor que 1.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=semilla)
    return next(splitter.split(df, y, groups))


def crear_cv(n_splits: int, semilla: int) -> StratifiedGroupKFold:
    """Validación cruzada estratificada y agrupada sobre el entrenamiento."""
    if n_splits < 2:
        raise ValueError("--cv-folds debe ser 0 (sin validación cruzada) o un entero >= 2.")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
