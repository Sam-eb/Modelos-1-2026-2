"""Entrenamiento reproducible del modelo Titanic Espacial."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score

from src.almacenamiento import guardar_version
from src.datos import cargar_datos_entrenamiento
from src.modelo import crear_modelo, crear_modelo_base
from src.preprocesamiento import FEATURES, preparar_datos


HIPERPARAMETROS_RF = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena y evalúa un Random Forest sobre Titanic Espacial, "
            "separando pasajeros por grupo de viaje."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Ruta al CSV de entrenamiento.",
    )
    parser.add_argument(
        "--target",
        default="Transported",
        help="Nombre de la columna objetivo (default: Transported).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se guardará la carpeta de la versión.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Identificador único para esta versión del modelo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Semilla para el modelo y las particiones aleatorias (default: 67).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de grupos reservada para prueba (default: 0.2).",
    )
    return parser


def _obtener_hash_commit() -> str | None:
    raiz_proyecto = Path(__file__).resolve().parent
    try:
        resultado = subprocess.run(
            ["git", "-C", str(raiz_proyecto), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return resultado.stdout.strip() or None


def _entrenar(args: argparse.Namespace) -> tuple[Path, Mapping[str, float]]:
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size debe ser un valor mayor que 0 y menor que 1.")

    df, y, groups = cargar_datos_entrenamiento(args.data, target=args.target)
    if not y.isin([0, 1]).all():
        raise ValueError(
            f"La variable objetivo '{args.target}' debe ser binaria "
            "y contener únicamente 0/1 o False/True."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.seed,
    )
    train_idx, test_idx = next(splitter.split(df, y, groups))

    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    X_train = preparar_datos(df_train)
    X_test = preparar_datos(df_test)

    modelo = crear_modelo(semilla=args.seed)
    modelo.fit(X_train, y_train)

    modelo_base = crear_modelo_base()
    modelo_base.fit(X_train, y_train)

    accuracy_test = accuracy_score(y_test, modelo.predict(X_test))
    accuracy_test_base = accuracy_score(y_test, modelo_base.predict(X_test))

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed,
    )
    scores_cv = cross_val_score(
        crear_modelo(semilla=args.seed),
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )

    metricas = {
        "accuracy_test": accuracy_test,
        "accuracy_test_base": accuracy_test_base,
        "accuracy_cv_mean": np.mean(scores_cv),
        "accuracy_cv_std": np.std(scores_cv),
    }

    hiperparametros = {
        nombre: modelo.named_steps["rf"].get_params()[nombre]
        for nombre in HIPERPARAMETROS_RF
    }
    metadata = {
        "fecha_creacion": datetime.now(timezone.utc).isoformat(),
        "tipo_problema": "clasificacion_binaria",
        "target": args.target,
        "algoritmo": "RandomForestClassifier",
        "hiperparametros": hiperparametros,
        "features": list(FEATURES),
        "tipos_features": {
            "HomePlanet": "categorica",
            "CryoSleep": "numerica_binaria",
            "Deck": "categorica",
            "Age": "numerica",
            "RoomService": "numerica",
            "Spa": "numerica",
            "VRDeck": "numerica",
        },
        "semilla": args.seed,
        "fuente_datos": str(args.data),
        "observaciones": {
            "total": len(df),
            "entrenamiento": len(train_idx),
            "prueba": len(test_idx),
        },
        "estrategia_separacion": {
            "metodo": "GroupShuffleSplit",
            "variable_agrupacion": "PassengerId (grupo de viaje)",
            "test_size": args.test_size,
        },
        "validacion_cruzada": {
            "metodo": "StratifiedGroupKFold",
            "n_splits": 5,
            "aplicada_sobre": "conjunto de entrenamiento",
        },
        "metrica": "accuracy",
        "version_python": sys.version,
        "requirements": "requirements.txt",
        "hash_commit": _obtener_hash_commit(),
    }

    directorio_version = guardar_version(
        modelo,
        metricas,
        metadata,
        args.output_dir,
        args.version,
    )
    return directorio_version, metricas


def main(argv: Sequence[str] | None = None) -> int:
    parser = crear_parser()
    args = parser.parse_args(argv)
    try:
        directorio_version, metricas = _entrenar(args)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"Accuracy Random Forest (test): {metricas['accuracy_test']:.4f}")
    print(f"Accuracy modelo base (test): {metricas['accuracy_test_base']:.4f}")
    print(
        f"Accuracy CV (media +/- std): "
        f"{metricas['accuracy_cv_mean']:.4f} +/- {metricas['accuracy_cv_std']:.4f}"
    )
    print(f"Versión guardada en: {directorio_version}")
    print("Archivos guardados: modelo.joblib, metrics.json, metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
