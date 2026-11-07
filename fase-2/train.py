"""Entrenamiento reproducible del modelo Titanic Espacial."""

import argparse
import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version as version_paquete
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from src.almacenamiento import guardar_version
from src.datos import cargar_datos_entrenamiento
from src.modelo import crear_modelo, crear_modelo_base
from src.particion import crear_cv, dividir_por_grupos
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
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help=(
            "Particiones de la validación cruzada agrupada sobre el entrenamiento; "
            "0 la omite (default: 5)."
        ),
    )
    return parser


# Archivos cuyo estado en git define el código usado para entrenar.
ARCHIVOS_CODIGO = ("src", "train.py", "predict.py", "requirements.txt")


def _ejecutar_git(*argumentos: str) -> str | None:
    raiz_proyecto = Path(__file__).resolve().parent
    try:
        resultado = subprocess.run(
            ["git", "-C", str(raiz_proyecto), *argumentos],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return resultado.stdout.strip()


def _info_codigo() -> dict[str, object]:
    """Commit de git usado para entrenar.

    Dentro de Docker no hay carpeta .git; en ese caso se usa la variable de
    entorno GIT_COMMIT (p. ej. `docker run -e GIT_COMMIT=$(git rev-parse HEAD)`).
    """
    commit = _ejecutar_git("rev-parse", "HEAD")
    if commit:
        cambios = _ejecutar_git("status", "--porcelain", "--", *ARCHIVOS_CODIGO)
        return {
            "hash_commit": commit,
            "cambios_sin_commit": bool(cambios),
            "origen": "git",
        }
    commit_entorno = os.environ.get("GIT_COMMIT") or None
    return {
        "hash_commit": commit_entorno,
        "cambios_sin_commit": None,
        "origen": "variable GIT_COMMIT" if commit_entorno else "no disponible",
    }


def _sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _entrenar(args: argparse.Namespace) -> tuple[Path, Mapping[str, float]]:
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size debe ser un valor mayor que 0 y menor que 1.")
    if args.cv_folds == 1 or args.cv_folds < 0:
        raise ValueError("--cv-folds debe ser 0 (sin validación cruzada) o un entero >= 2.")
    destino = Path(args.output_dir) / args.version
    if destino.exists():
        raise FileExistsError(
            f"La versión '{args.version}' ya existe en {destino}; use otro --version."
        )

    df, y, groups = cargar_datos_entrenamiento(args.data, target=args.target)
    if not y.isin([0, 1]).all():
        raise ValueError(
            f"La variable objetivo '{args.target}' debe ser binaria "
            "y contener únicamente 0/1 o False/True."
        )

    train_idx, test_idx = dividir_por_grupos(df, y, groups, args.test_size, args.seed)

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

    metricas: dict[str, float | None] = {
        "accuracy_test": accuracy_test,
        "accuracy_test_base": accuracy_test_base,
        "accuracy_cv_mean": None,
        "accuracy_cv_std": None,
    }
    if args.cv_folds:
        # El pipeline completo se reajusta en cada fold: sin fuga entre folds.
        cv = crear_cv(args.cv_folds, args.seed)
        scores_cv = cross_val_score(
            crear_modelo(semilla=args.seed),
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )
        metricas["accuracy_cv_mean"] = float(np.mean(scores_cv))
        metricas["accuracy_cv_std"] = float(np.std(scores_cv))

    hiperparametros = {
        nombre: modelo.named_steps["rf"].get_params()[nombre]
        for nombre in HIPERPARAMETROS_RF
    }
    ruta_datos = Path(args.data)
    metadata = {
        "version": args.version,
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
        "datos": {
            "fuente": "Kaggle - Spaceship Titanic (train.csv)",
            "archivo": ruta_datos.as_posix(),
            "sha256": _sha256(ruta_datos),
        },
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
        "validacion_cruzada": (
            {
                "metodo": "StratifiedGroupKFold",
                "n_splits": args.cv_folds,
                "aplicada_sobre": "conjunto de entrenamiento",
            }
            if args.cv_folds
            else None
        ),
        "metrica": "accuracy",
        "etiquetas": {"1": "Transportado (Transported=True)", "0": "No transportado"},
        "version_python": platform.python_version(),
        "librerias": {
            nombre: version_paquete(nombre)
            for nombre in ("scikit-learn", "pandas", "numpy", "joblib")
        },
        "requirements": "requirements.txt",
        "codigo": _info_codigo(),
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
    if metricas["accuracy_cv_mean"] is not None:
        print(
            f"Accuracy CV (media +/- std): "
            f"{metricas['accuracy_cv_mean']:.4f} +/- {metricas['accuracy_cv_std']:.4f}"
        )
    print(f"Versión guardada en: {directorio_version}")
    print("Archivos guardados: modelo.joblib, metrics.json, metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
