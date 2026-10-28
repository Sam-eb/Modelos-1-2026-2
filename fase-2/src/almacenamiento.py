"""Persistencia de versiones completas del modelo."""

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ARCHIVOS_VERSION = ("modelo.joblib", "metrics.json", "metadata.json")
_VERSION_SEGURA = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def guardar_version(
    modelo: Any,
    metricas: Mapping[str, Any],
    metadata: Mapping[str, Any],
    output_dir: str | Path,
    version: str,
) -> Path:
    """Guarda una versión completa y no permite reemplazar una ya existente.

    Los archivos se preparan en un directorio temporal antes de reservar el
    nombre final. Si falla la escritura, se limpian los archivos temporales y
    la carpeta de versión reservada.
    """
    if not isinstance(version, str) or not _VERSION_SEGURA.fullmatch(version):
        raise ValueError(
            "La versión debe comenzar con una letra o número y solo contener "
            "letras, números, puntos, guiones o guiones bajos."
        )
    if not isinstance(metricas, Mapping):
        raise TypeError("Las métricas deben proporcionarse como un mapping.")
    if not isinstance(metadata, Mapping):
        raise TypeError("Los metadatos deben proporcionarse como un mapping.")

    directorio_base = Path(output_dir)
    directorio_base.mkdir(parents=True, exist_ok=True)
    destino = directorio_base / version
    if destino.exists() or destino.is_symlink():
        raise FileExistsError(f"La versión '{version}' ya existe en {destino}.")

    directorio_temporal = Path(
        tempfile.mkdtemp(prefix=f".{version}.", dir=directorio_base)
    )
    try:
        joblib.dump(modelo, directorio_temporal / "modelo.joblib")
        _guardar_json(dict(metricas), directorio_temporal / "metrics.json")
        _guardar_json(dict(metadata), directorio_temporal / "metadata.json")
        try:
            destino.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"La versión '{version}' ya existe en {destino}."
            ) from error
        try:
            for nombre in ARCHIVOS_VERSION:
                os.rename(
                    directorio_temporal / nombre,
                    destino / nombre,
                )
        except Exception:
            shutil.rmtree(destino)
            raise
    except Exception:
        if directorio_temporal.exists():
            shutil.rmtree(directorio_temporal)
        raise

    return destino


def cargar_version(
    directorio_version: str | Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Carga el modelo, las métricas y los metadatos de una versión guardada.

    Solo deben cargarse archivos joblib de fuentes confiables, ya que el formato
    puede ejecutar código durante la deserialización.
    """
    directorio = Path(directorio_version)
    rutas = {nombre: directorio / nombre for nombre in ARCHIVOS_VERSION}
    faltantes = [ruta.name for ruta in rutas.values() if not ruta.is_file()]
    if faltantes:
        raise FileNotFoundError(
            f"La versión '{directorio}' está incompleta; faltan: "
            f"{', '.join(faltantes)}."
        )

    modelo = joblib.load(rutas["modelo.joblib"])
    metricas = _cargar_json(rutas["metrics.json"])
    metadata = _cargar_json(rutas["metadata.json"])
    return modelo, metricas, metadata


def _guardar_json(datos: dict[str, Any], ruta: Path) -> None:
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(
            datos,
            archivo,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_convertir_numpy,
        )
        archivo.write("\n")


def _convertir_numpy(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"El objeto de tipo {type(obj).__name__} no es serializable en JSON.")


def _cargar_json(ruta: Path) -> dict[str, Any]:
    try:
        with ruta.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
    except json.JSONDecodeError as error:
        raise ValueError(f"El archivo '{ruta}' no contiene JSON válido.") from error
    if not isinstance(datos, dict):
        raise ValueError(f"El contenido de '{ruta}' debe ser un objeto JSON.")
    return datos
