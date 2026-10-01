"""Preprocesamiento del Titanic Espacial (Fase 1).

Reproduce las reglas de imputación del notebook, ajustándose solo con los
datos de entrenamiento:
  - HomePlanet y Deck: moda.
  - Gastos de pasajeros con CryoSleep=True: 0 (dormidos no gastan).
  - CryoSleep faltante con algún gasto > 0: 0 (quien gasta no está dormido).
  - CryoSleep restante: moda.
  - Age: muestreo de una normal(media, desviación) con valores >= 0.
  - Gastos restantes: mediana.
Luego: log1p sobre los gastos y one-hot para HomePlanet y Deck.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

FEATURES = ["HomePlanet", "CryoSleep", "Deck", "Age", "RoomService", "Spa", "VRDeck"]
GASTOS = ["RoomService", "Spa", "VRDeck"]


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Datos crudos de Kaggle -> las 7 variables de entrada del modelo.

    Deriva Deck desde Cabin y convierte CryoSleep a 0/1 (los NaN se conservan;
    los imputa el pipeline). No calcula nada a partir de los datos.
    """
    df = df.copy()
    if "Deck" not in df.columns:
        df["Deck"] = df["Cabin"].str.split("/", expand=True)[0]
    df["CryoSleep"] = pd.to_numeric(df["CryoSleep"])
    return df[FEATURES]


class ImputadorTitanic(BaseEstimator, TransformerMixin):
    """Imputación con las reglas del notebook. fit usa solo entrenamiento."""

    def __init__(self, semilla=67):
        self.semilla = semilla

    def _reglas_dominio(self, X):
        """Reglas fijas de CryoSleep y gastos (no dependen de los datos)."""
        X = X.copy()
        dormido = X["CryoSleep"] == True  # noqa: E712 (NaN no cuenta como dormido)
        for g in GASTOS:
            X.loc[dormido, g] = X.loc[dormido, g].fillna(0)
        gasta = X["CryoSleep"].isna() & (X[GASTOS].sum(axis=1, skipna=True) > 0)
        X.loc[gasta, "CryoSleep"] = 0
        return X

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.moda_homeplanet_ = X["HomePlanet"].mode()[0]
        self.moda_deck_ = X["Deck"].mode()[0]
        Xr = self._reglas_dominio(X)
        self.moda_cryosleep_ = Xr["CryoSleep"].mode()[0]
        self.media_age_ = X["Age"].mean()
        self.std_age_ = X["Age"].std()
        self.medianas_gastos_ = {g: Xr[g].median() for g in GASTOS}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["HomePlanet"] = X["HomePlanet"].fillna(self.moda_homeplanet_)
        X["Deck"] = X["Deck"].fillna(self.moda_deck_)
        X = self._reglas_dominio(X)
        X["CryoSleep"] = X["CryoSleep"].fillna(self.moda_cryosleep_)
        # Age: muestreo normal reproducible (misma semilla en cada llamada)
        faltan = X["Age"].isna()
        if faltan.any():
            rng = np.random.RandomState(self.semilla)
            muestra = rng.normal(self.media_age_, self.std_age_, faltan.sum())
            X.loc[faltan, "Age"] = np.clip(muestra, 0, None)
        for g in GASTOS:
            X[g] = X[g].fillna(self.medianas_gastos_[g])
        return X


def crear_preprocesamiento(semilla=67) -> "Pipeline":
    """Imputación del notebook + log1p en gastos + one-hot en categóricas."""
    from sklearn.pipeline import Pipeline
    codificar = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["HomePlanet", "Deck"]),
            ("log_gastos", FunctionTransformer(np.log1p), GASTOS),
        ],
        remainder="passthrough",  # CryoSleep y Age quedan tal cual
    )
    return Pipeline([("imputar", ImputadorTitanic(semilla)), ("codificar", codificar)])