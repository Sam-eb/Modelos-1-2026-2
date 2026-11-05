"""Carga y validación de datos para el Titanic Espacial."""

from pathlib import Path

import pandas as pd

from .preprocesamiento import convertir_booleano

COLUMNAS_ENTRADA_MINIMAS = {
    "HomePlanet",
    "CryoSleep",
    "Age",
    "RoomService",
    "Spa",
    "VRDeck",
}
COLUMNAS_CABINA = {"Cabin", "Deck"}
COLUMNAS_NUMERICAS = ["Age", "RoomService", "Spa", "VRDeck"]


def cargar_csv(ruta: str | Path) -> pd.DataFrame:
    """Carga un archivo CSV validando su existencia y contenido."""
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise FileNotFoundError(f"El archivo de datos no existe: {ruta_path}")
    if not ruta_path.is_file():
        raise ValueError(f"La ruta especificada no es un archivo válido: {ruta_path}")

    try:
        df = pd.read_csv(ruta_path)
    except Exception as error:
        raise ValueError(f"Error al leer el archivo CSV '{ruta_path}': {error}") from error

    if df.empty:
        raise ValueError(f"El archivo CSV '{ruta_path}' está vacío (0 filas).")

    return df


def validar_columnas_entrada(df: pd.DataFrame, es_entrenamiento: bool = False, target: str | None = None) -> None:
    """Valida la presencia de las columnas requeridas para el modelo."""
    columnas_presentes = set(df.columns)

    # Verificar columnas básicas de características
    faltantes = COLUMNAS_ENTRADA_MINIMAS - columnas_presentes
    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en los datos: {sorted(faltantes)}. "
            f"Columnas disponibles: {sorted(columnas_presentes)}"
        )

    # Verificar que exista al menos 'Cabin' o 'Deck'
    if not (COLUMNAS_CABINA & columnas_presentes):
        raise ValueError(
            "Se requiere al menos la columna 'Cabin' o 'Deck' para derivar la cubierta. "
            f"Columnas disponibles: {sorted(columnas_presentes)}"
        )

    # En entrenamiento, se requiere PassengerId (para agrupamiento) y la variable objetivo
    if es_entrenamiento:
        if "PassengerId" not in columnas_presentes:
            raise ValueError(
                "La columna 'PassengerId' es requerida en entrenamiento para la separación por grupos. "
                f"Columnas disponibles: {sorted(columnas_presentes)}"
            )
        if target and target not in columnas_presentes:
            raise ValueError(
                f"La columna objetivo '{target}' no se encuentra en los datos de entrenamiento. "
                f"Columnas disponibles: {sorted(columnas_presentes)}"
            )


def validar_tipos_datos(df: pd.DataFrame) -> None:
    """Valida que los valores no nulos sean compatibles con el modelo.

    - Age y gastos: numéricos y no negativos.
    - CryoSleep: True/False, 1/0 o su equivalente en texto.
    Los nulos son válidos: los trata la imputación del pipeline.
    """
    for col in COLUMNAS_NUMERICAS:
        if col in df.columns:
            valores_originales = df[col].dropna()
            valores_coercionados = pd.to_numeric(valores_originales, errors="coerce")
            corruptos = valores_originales[valores_coercionados.isna()]
            if not corruptos.empty:
                ejemplos = corruptos.head(3).tolist()
                raise ValueError(
                    f"La columna '{col}' contiene valores no numéricos no válidos: {ejemplos}"
                )
            negativos = valores_originales[valores_coercionados < 0]
            if not negativos.empty:
                raise ValueError(
                    f"La columna '{col}' contiene valores negativos: "
                    f"{negativos.head(3).tolist()}"
                )

    if "CryoSleep" in df.columns:
        convertir_booleano(df["CryoSleep"], nombre="CryoSleep")


def cargar_datos_entrenamiento(
    ruta: str | Path,
    target: str = "Transported",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carga y valida el conjunto de datos para entrenamiento.

    Retorna:
        df: DataFrame con las características y columnas originales.
        y: Serie con la variable objetivo convertida a int (0 o 1).
        groups: Serie con los identificadores de grupo extraídos de PassengerId.
    """
    df = cargar_csv(ruta)
    validar_columnas_entrada(df, es_entrenamiento=True, target=target)
    validar_tipos_datos(df)

    # Validar que target no tenga nulos en entrenamiento
    y_raw = df[target]
    if y_raw.isna().any():
        n_nulos = y_raw.isna().sum()
        raise ValueError(
            f"La variable objetivo '{target}' contiene {n_nulos} valores nulos en el conjunto de entrenamiento."
        )

    # Convertir target a int (0 o 1); acepta True/False, 1/0 o su texto
    y = convertir_booleano(y_raw, nombre=target).astype(int)

    # Extraer vector de grupo desde PassengerId (formato 'gggg_nn')
    groups = df["PassengerId"].astype(str).str.split("_").str[0]

    return df, y, groups


def cargar_datos_prediccion(ruta: str | Path) -> pd.DataFrame:
    """Carga y valida un conjunto de datos para generar predicciones."""
    df = cargar_csv(ruta)
    validar_columnas_entrada(df, es_entrenamiento=False)
    validar_tipos_datos(df)
    return df
