"""Preprocesamiento del Titanic Espacial para entrenamiento y predicción.

`preparar_datos` solo selecciona y convierte tipos (no aprende nada de los
datos). Todo lo que aprende de los datos (modas, medianas, categorías) vive en
el pipeline de `crear_preprocesamiento` y se ajusta únicamente con el
conjunto de entrenamiento.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

FEATURES = ["HomePlanet", "CryoSleep", "Deck", "Age", "RoomService", "Spa", "VRDeck"]
CATEGORICAS = ["HomePlanet", "Deck"]
NUMERICAS = ["Age", "RoomService", "Spa", "VRDeck"]
GASTOS = ["RoomService", "Spa", "VRDeck"]

# Representaciones aceptadas para CryoSleep (se comparan en minúsculas).
_VALORES_BOOLEANOS = {
    "true": 1.0,
    "false": 0.0,
    "1": 1.0,
    "0": 0.0,
    "1.0": 1.0,
    "0.0": 0.0,
}


def convertir_booleano(serie: pd.Series, nombre: str = "CryoSleep") -> pd.Series:
    """Convierte True/False, 1/0 o sus equivalentes en texto a 1.0/0.0.

    Los nulos se conservan como NaN (los imputa el pipeline). Cualquier otro
    valor produce un ValueError.
    """
    texto = serie.astype("string").str.strip().str.lower()
    convertida = texto.map(_VALORES_BOOLEANOS)
    invalidos = serie[texto.notna() & convertida.isna()]
    if not invalidos.empty:
        raise ValueError(
            f"La columna '{nombre}' contiene valores no convertibles a booleano: "
            f"{invalidos.head(3).tolist()}"
        )
    return convertida.astype(float)


def _a_texto(serie: pd.Series) -> pd.Series:
    """Convierte a texto conservando los nulos como NaN (aunque la columna
    venga completamente vacía y pandas la haya leído como numérica)."""
    texto = serie.astype("string").str.strip()
    return texto.astype(object).where(texto.notna() & (texto != ""), np.nan)


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Datos crudos -> las 7 variables de entrada, en el orden de FEATURES.

    Deriva Deck desde Cabin (formato 'Deck/Num/Side') si Deck no viene dado.
    Las columnas adicionales se ignoran.
    """
    datos = pd.DataFrame(index=df.index)
    datos["HomePlanet"] = _a_texto(df["HomePlanet"])
    datos["CryoSleep"] = convertir_booleano(df["CryoSleep"])
    if "Deck" in df.columns:
        datos["Deck"] = _a_texto(df["Deck"])
    else:
        datos["Deck"] = _a_texto(_a_texto(df["Cabin"]).str.split("/").str[0])
    for columna in NUMERICAS:
        datos[columna] = pd.to_numeric(df[columna]).astype(float)
    return datos[FEATURES]


class ImputadorTitanic(BaseEstimator, TransformerMixin):
    """Imputa con estadísticas aprendidas en entrenamiento y reglas de dominio.

    - HomePlanet y Deck: moda.
    - Gastos de pasajeros con CryoSleep=1: 0 (dormidos no gastan).
    - CryoSleep faltante con algún gasto > 0: 0 (quien gasta no está dormido).
    - CryoSleep restante: moda.
    - Age: mediana (determinista; corrige la imputación aleatoria de la Fase 1).
    - Gastos restantes: mediana.
    """

    @staticmethod
    def _reglas_dominio(X: pd.DataFrame) -> pd.DataFrame:
        datos = X.copy()
        dormido = datos["CryoSleep"] == 1
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


def crear_preprocesamiento() -> Pipeline:
    """Crea el pipeline de imputación, log1p y codificación one-hot.

    Las categorías no vistas en entrenamiento (HomePlanet o Deck) se codifican
    con todas las columnas one-hot en 0 (handle_unknown="ignore").
    """
    codificar = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAS,
            ),
            ("log_gastos", FunctionTransformer(np.log1p), GASTOS),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("imputar", ImputadorTitanic()),
            ("codificar", codificar),
        ]
    )
