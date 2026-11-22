"""Adaptador entre la API (Fase 3) y el código de la Fase 2.

Es el único módulo de `fase-3` que importa `fase-2`. La carpeta `fase-2` se
conserva congelada: aquí se reutilizan sus funciones tal como están, sin copiar
su lógica (validación, preprocesamiento, pipeline guardado, entrenamiento y
almacenamiento de versiones).

Dos de las funciones reutilizadas son internas de los scripts de la Fase 2
(`train._entrenar` y `predict._generar_predicciones`). Se usan a conciencia:
como `fase-2` no cambia, su comportamiento es estable, y si algún día cambiara
solo habría que ajustar este módulo.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

DIR_FASE3 = Path(__file__).resolve().parent
DIR_FASE2 = DIR_FASE3.parent / "fase-2"

# Hace importables `src`, `train` y `predict` de la Fase 2. Es el único efecto
# al importar este módulo; no carga datos ni modelos.
if str(DIR_FASE2) not in sys.path:
    sys.path.insert(0, str(DIR_FASE2))

import predict as _predict_fase2  # noqa: E402
import train as _train_fase2  # noqa: E402
from src.almacenamiento import ARCHIVOS_VERSION, cargar_version  # noqa: E402
from src.datos import validar_columnas_entrada, validar_tipos_datos  # noqa: E402
from src.preprocesamiento import FEATURES  # noqa: E402

_PATRON_VERSION = re.compile(r"v(\d+)\Z")
INTENTOS_ENTRENAMIENTO = 3


class ModeloNoDisponible(RuntimeError):
    """La versión configurada no existe, está incompleta o no es compatible."""


@dataclass(frozen=True)
class ModeloCargado:
    """Una versión guardada lista para predecir."""

    version: str
    directorio: Path
    modelo: Any = field(repr=False)
    metricas: dict[str, Any] = field(repr=False)
    metadata: dict[str, Any] = field(repr=False)


def versiones_disponibles(directorio_modelos: str | Path) -> list[str]:
    """Versiones completas (con sus tres archivos) en `directorio_modelos`."""
    base = Path(directorio_modelos)
    if not base.is_dir():
        return []
    return sorted(
        hijo.name
        for hijo in base.iterdir()
        if hijo.is_dir()
        and not hijo.name.startswith(".")
        and all((hijo / nombre).is_file() for nombre in ARCHIVOS_VERSION)
    )


def cargar_modelo(directorio_modelos: str | Path, version: str) -> ModeloCargado:
    """Carga una versión y comprueba que use las variables de este código.

    Lanza ModeloNoDisponible con un mensaje que indica qué versiones hay.
    """
    base = Path(directorio_modelos)
    if not version or Path(version).name != version:
        raise ModeloNoDisponible(f"Identificador de versión inválido: {version!r}.")
    directorio = base / version
    disponibles = versiones_disponibles(base)
    if version not in disponibles:
        raise ModeloNoDisponible(
            f"La versión '{version}' no está disponible en el directorio de "
            f"modelos configurado (MODELOS_DIR). "
            f"Versiones disponibles: {disponibles or 'ninguna'}. "
            "Elija una con la variable de entorno MODELO_VERSION."
        )
    try:
        modelo, metricas, metadata = cargar_version(directorio)
    except (FileNotFoundError, ValueError) as error:
        raise ModeloNoDisponible(str(error)) from error

    features_version = metadata.get("features")
    if features_version != list(FEATURES):
        raise ModeloNoDisponible(
            f"La versión '{version}' espera las variables {features_version}, "
            f"pero este código usa {list(FEATURES)}; no son compatibles."
        )
    return ModeloCargado(version, directorio, modelo, metricas, metadata)


def predecir(
    modelo: ModeloCargado,
    registros: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Predice uno o varios registros con la validación y el pipeline de la Fase 2.

    Los registros se convierten en una tabla (una fila por registro, en el
    mismo orden). Los `None` se tratan como datos faltantes y los imputa el
    pipeline guardado. Lanza ValueError si falta una columna requerida o un
    valor no es compatible, con un mensaje que lo identifica.
    """
    if not registros:
        raise ValueError("No se recibieron registros para predecir.")
    datos = pd.DataFrame([dict(registro) for registro in registros])
    validar_columnas_entrada(datos, es_entrenamiento=False)
    validar_tipos_datos(datos)
    return _predict_fase2._generar_predicciones(modelo.modelo, datos)


def siguiente_version(directorio_modelos: str | Path) -> str:
    """Siguiente identificador libre con la forma vN (v3, v4, ...)."""
    base = Path(directorio_modelos)
    existentes = {hijo.name for hijo in base.iterdir()} if base.is_dir() else set()
    numeros = [
        int(coincidencia.group(1))
        for nombre in existentes
        if (coincidencia := _PATRON_VERSION.match(nombre))
    ]
    numero = max(numeros, default=0) + 1
    while f"v{numero}" in existentes:
        numero += 1
    return f"v{numero}"


def entrenar_nueva_version(
    datos: str | Path,
    directorio_modelos: str | Path,
    target: str = "Transported",
    semilla: int = 42,
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> tuple[str, Path, dict[str, Any]]:
    """Entrena con la función de `train.py` y guarda una versión nueva.

    Usa el parser de `train.py`, de modo que los valores se interpretan igual
    que en la línea de comandos. Nunca sobrescribe: si otra solicitud ocupó el
    identificador al mismo tiempo, reintenta con el siguiente.
    """
    ultimo_error: FileExistsError | None = None
    for _ in range(INTENTOS_ENTRENAMIENTO):
        version = siguiente_version(directorio_modelos)
        args = _train_fase2.crear_parser().parse_args([
            "--data", str(datos),
            "--target", target,
            "--output-dir", str(directorio_modelos),
            "--version", version,
            "--seed", str(semilla),
            "--test-size", str(test_size),
            "--cv-folds", str(cv_folds),
        ])
        try:
            directorio, metricas = _train_fase2._entrenar(args)
        except FileExistsError as error:
            ultimo_error = error
            continue
        return version, directorio, dict(metricas)
    raise FileExistsError(
        f"No se encontró un identificador libre tras {INTENTOS_ENTRENAMIENTO} intentos."
    ) from ultimo_error
