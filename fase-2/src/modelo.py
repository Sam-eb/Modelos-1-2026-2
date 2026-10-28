"""Construcción de los modelos de Titanic Espacial."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocesamiento import crear_preprocesamiento


def crear_modelo(semilla: int = 67) -> Pipeline:
    """Construye el pipeline completo con preprocesamiento y Random Forest."""
    return Pipeline(
        [
            ("preprocesamiento", crear_preprocesamiento(semilla=semilla)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=16,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=semilla,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def crear_modelo_base() -> DummyClassifier:
    """Construye el clasificador base que predice la clase más frecuente."""
    return DummyClassifier(strategy="most_frequent")
