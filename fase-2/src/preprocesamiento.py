"""Preprocesamiento del Titanic Espacial para entrenamiento y predicción."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

FEATURES = ["HomePlanet", "CryoSleep", "Deck", "Age", "RoomService", "Spa", "VRDeck"]
GASTOS = ["RoomService", "Spa", "VRDeck"]


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    datos = df.copy()
    if "Deck" not in datos.columns:
        datos["Deck"] = datos["Cabin"].str.split("/", expand=True)[0]
    datos["CryoSleep"] = pd.to_numeric(datos["CryoSleep"])
    datos["Age"] = pd.to_numeric(datos["Age"])
    for gasto in GASTOS:
        datos[gasto] = pd.to_numeric(datos[gasto])
    return datos[FEATURES]


class ImputadorTitanic(BaseEstimator, TransformerMixin):
    """Imputa con estadísticas de entrenamiento y las reglas de dominio."""

    def __init__(self, semilla: int = 67):
        self.semilla = semilla

    @staticmethod
    def _reglas_dominio(X: pd.DataFrame) -> pd.DataFrame:
        datos = X.copy()
        dormido = datos["CryoSleep"] == True  # noqa: E712
        for gasto in GASTOS:
            datos.loc[dormido, gasto] = datos.loc[dormido, gasto].fillna(0)
        gasta = datos["CryoSleep"].isna() & (
            datos[GASTOS].sum(axis=1, skipna=True) > 0
        )
        datos.loc[gasta, "CryoSleep"] = 0
        return datos

    def fit(self, X: pd.DataFrame, y=None):
        datos = pd.DataFrame(X).copy()
        self.moda_homeplanet_ = datos["HomePlanet"].mode()[0]
        self.moda_deck_ = datos["Deck"].mode()[0]

        datos = self._reglas_dominio(datos)
        self.moda_cryosleep_ = datos["CryoSleep"].mode()[0]
        self.mediana_age_ = datos["Age"].median()
        self.medianas_gastos_ = {
            gasto: datos[gasto].median() for gasto in GASTOS
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        datos = pd.DataFrame(X).copy()
        datos["HomePlanet"] = datos["HomePlanet"].fillna(self.moda_homeplanet_)
        datos["Deck"] = datos["Deck"].fillna(self.moda_deck_)
        datos = self._reglas_dominio(datos)
        datos["CryoSleep"] = datos["CryoSleep"].fillna(self.moda_cryosleep_)
        datos["Age"] = datos["Age"].fillna(self.mediana_age_)
        for gasto in GASTOS:
            datos[gasto] = datos[gasto].fillna(self.medianas_gastos_[gasto])
        return datos


def crear_preprocesamiento(semilla: int = 67) -> Pipeline:
    """Crea el pipeline de imputación, log1p y codificación one-hot."""
    codificar = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["HomePlanet", "Deck"],
            ),
            ("log_gastos", FunctionTransformer(np.log1p), GASTOS),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("imputar", ImputadorTitanic(semilla=semilla)),
            ("codificar", codificar),
        ]
    )
