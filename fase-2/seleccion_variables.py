"""Verificación de la selección de variables de la Fase 1 sin usar datos de prueba.

En la Fase 1 las asociaciones con `Transported` se calcularon con todo el
dataset, antes de separar entrenamiento y prueba. Este script repite los
mismos criterios (|r| de Pearson / punto-biserial >= 0,2 para numéricas y
binarias; V de Cramér >= 0,2 para categóricas) usando únicamente:

- la partición de entrenamiento de `train.py` (misma semilla y test-size), y
- la parte de entrenamiento de cada fold de su validación cruzada.

El conjunto de prueba nunca se usa. El resultado no modifica el modelo: solo
informa si la decisión de la Fase 1 se sostiene sin la intervención de la
prueba. La lista de variables del modelo está fija en `src/preprocesamiento.py`.
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats.contingency import association

from src.datos import cargar_datos_entrenamiento
from src.particion import crear_cv, dividir_por_grupos
from src.preprocesamiento import FEATURES, convertir_booleano

UMBRAL = 0.2
# Margen bajo el cual una variable se reporta como "cercana al umbral".
MARGEN_CERCANIA = 0.02

# Candidatas analizadas en el EDA de la Fase 1.
CANDIDATAS_NUMERICAS = [
    "CryoSleep",
    "VIP",
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Cabin_Num",
]
CANDIDATAS_CATEGORICAS = ["HomePlanet", "Destination", "Deck", "Side", "Group_Size"]

# Decisión de la Fase 1: variables que superaron el umbral y variables
# conservadas por criterio explícito aunque no lo superaran.
SELECCION_POR_UMBRAL_FASE1 = {"CryoSleep", "RoomService", "Spa", "Deck", "VRDeck"}
CONSERVADAS_POR_CRITERIO_FASE1 = {"HomePlanet", "Age"}


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recalcula los criterios de selección de variables de la Fase 1 "
            "usando solo la partición de entrenamiento y cada fold de la CV."
        )
    )
    parser.add_argument("--data", required=True, help="Ruta al CSV de entrenamiento.")
    parser.add_argument(
        "--target",
        default="Transported",
        help="Nombre de la columna objetivo (default: Transported).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Semilla de la partición, igual a la de train.py (default: 67).",
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
        help="Folds de la validación cruzada agrupada (default: 5).",
    )
    parser.add_argument(
        "--output",
        default="outputs/seleccion_variables.csv",
        help="CSV donde se guarda la tabla (default: outputs/seleccion_variables.csv).",
    )
    return parser


def derivar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las candidatas de la Fase 1 a partir de los datos crudos.

    Solo son transformaciones fila a fila (no aprenden de los datos), salvo
    Group_Size, que cuenta los pasajeros del propio grupo; como la partición
    es por grupos, ese conteo es idéntico dentro de cualquier partición.
    """
    cabina = df["Cabin"].astype("string").str.split("/", expand=True)
    grupo = df["PassengerId"].astype(str).str.split("_").str[0]
    return pd.DataFrame(
        {
            "CryoSleep": convertir_booleano(df["CryoSleep"], "CryoSleep"),
            "VIP": convertir_booleano(df["VIP"], "VIP"),
            "Age": pd.to_numeric(df["Age"]),
            "RoomService": pd.to_numeric(df["RoomService"]),
            "FoodCourt": pd.to_numeric(df["FoodCourt"]),
            "ShoppingMall": pd.to_numeric(df["ShoppingMall"]),
            "Spa": pd.to_numeric(df["Spa"]),
            "VRDeck": pd.to_numeric(df["VRDeck"]),
            "Cabin_Num": pd.to_numeric(cabina[1]),
            "HomePlanet": df["HomePlanet"],
            "Destination": df["Destination"],
            "Deck": cabina[0],
            "Side": cabina[2],
            "Group_Size": grupo.map(grupo.value_counts()),
        },
        index=df.index,
    )


def calcular_asociaciones(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """|r| de Pearson para numéricas/binarias y V de Cramér para categóricas.

    Los nulos se excluyen par a par, como en la Fase 1.
    """
    valores = {col: X[col].astype(float).corr(y.astype(float)) for col in CANDIDATAS_NUMERICAS}
    for col in CANDIDATAS_CATEGORICAS:
        tabla = pd.crosstab(X[col], y)
        valores[col] = association(tabla, method="cramer")
    return pd.Series(valores)


def construir_tabla(
    X: pd.DataFrame,
    y: pd.Series,
    particiones: dict[str, np.ndarray],
) -> pd.DataFrame:
    tabla = pd.DataFrame(
        {nombre: calcular_asociaciones(X.iloc[idx], y.iloc[idx]) for nombre, idx in particiones.items()}
    )
    absolutos = tabla.abs()
    n_supera = (absolutos >= UMBRAL).sum(axis=1)

    resumen = pd.DataFrame(index=tabla.index)
    resumen["medida"] = [
        "V de Cramér" if col in CANDIDATAS_CATEGORICAS else "r de Pearson" for col in tabla.index
    ]
    resumen["min_abs"] = absolutos.min(axis=1)
    resumen["max_abs"] = absolutos.max(axis=1)
    resumen["distancia_umbral"] = (absolutos - UMBRAL).abs().min(axis=1)
    resumen["supera_umbral"] = [
        "todas" if n == len(particiones) else "ninguna" if n == 0 else f"{n} de {len(particiones)}"
        for n in n_supera
    ]
    resumen["decision_fase1"] = [
        "umbral" if col in SELECCION_POR_UMBRAL_FASE1
        else "criterio explícito" if col in CONSERVADAS_POR_CRITERIO_FASE1
        else "descartada"
        for col in tabla.index
    ]
    resumen["en_modelo"] = tabla.index.isin(FEATURES)
    resultado = pd.concat([resumen, tabla], axis=1)
    return resultado.sort_values("max_abs", ascending=False)


def _conclusiones(tabla: pd.DataFrame, n_particiones: int) -> list[str]:
    """Resume la comparación con la Fase 1 sin suavizar los casos límite."""
    supera = tabla["supera_umbral"]
    alguna_vez = set(tabla.index[supera != "ninguna"])
    descartadas = tabla[tabla["decision_fase1"] == "descartada"]
    descartadas_que_superan = sorted(set(descartadas.index) & alguna_vez)
    umbral_inestables = {
        col: supera[col] for col in sorted(SELECCION_POR_UMBRAL_FASE1) if supera[col] != "todas"
    }
    criterio_que_superan = {
        col: supera[col] for col in sorted(CONSERVADAS_POR_CRITERIO_FASE1) if supera[col] != "ninguna"
    }
    cercanas = list(tabla.index[tabla["distancia_umbral"] < MARGEN_CERCANIA])

    lineas = [
        f"Particiones evaluadas (sin datos de prueba): {n_particiones}",
        "1) Descartadas en la Fase 1 que superan el umbral en alguna partición: "
        f"{descartadas_que_superan or 'ninguna'} "
        f"(máximo entre descartadas: {descartadas['max_abs'].max():.3f}).",
    ]
    if umbral_inestables:
        lineas.append(
            "2) Seleccionadas por umbral en la Fase 1 que NO lo superan en todas las "
            f"particiones: {umbral_inestables}."
        )
    else:
        lineas.append(
            "2) Las seleccionadas por umbral en la Fase 1 lo superan en todas las particiones."
        )
    lineas.append(
        "3) Conservadas por criterio explícito en la Fase 1 (no por umbral): "
        f"{sorted(CONSERVADAS_POR_CRITERIO_FASE1)}; superan el umbral en: "
        f"{criterio_que_superan or 'ninguna partición'}."
    )
    if cercanas:
        lineas.append(
            f"4) Cercanas al umbral (a menos de {MARGEN_CERCANIA} en alguna partición): {cercanas}."
        )
    if not descartadas_que_superan and alguna_vez <= set(FEATURES):
        lineas.append(
            "Conclusión: ninguna variable fuera del modelo alcanza el umbral sin usar la "
            "prueba; las diferencias se concentran en variables límite que el modelo ya incluye."
        )
    else:
        lineas.append(
            "Conclusión: hay variables fuera del modelo que alcanzan el umbral sin usar la "
            "prueba; revisar la lista FEATURES."
        )
    return lineas


def _ejecutar(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], Path]:
    if args.cv_folds:
        cv = crear_cv(args.cv_folds, args.seed)
    df, y, groups = cargar_datos_entrenamiento(args.data, target=args.target)
    X = derivar_variables(df)

    train_idx, _ = dividir_por_grupos(df, y, groups, args.test_size, args.seed)
    particiones = {"entrenamiento": train_idx}
    if args.cv_folds:
        folds = cv.split(df.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx])
        for numero, (fold_train, _) in enumerate(folds, start=1):
            particiones[f"fold_{numero}"] = train_idx[fold_train]

    tabla = construir_tabla(X, y, particiones)
    ruta_salida = Path(args.output)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(ruta_salida, index_label="variable")
    return tabla, _conclusiones(tabla, len(particiones)), ruta_salida


def main(argv: Sequence[str] | None = None) -> int:
    parser = crear_parser()
    args = parser.parse_args(argv)
    try:
        tabla, conclusiones, ruta_salida = _ejecutar(args)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(tabla.round(3).to_string())
    print()
    for linea in conclusiones:
        print(linea)
    print(f"Tabla guardada en: {ruta_salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
