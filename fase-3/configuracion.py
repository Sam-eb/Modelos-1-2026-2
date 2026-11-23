"""Configuración de la API a partir de variables de entorno.

Todas las variables son opcionales. Los valores por defecto reproducen la
configuración de la Fase 2 y apuntan a las carpetas del repositorio, de modo
que la API funciona igual desde `fase-3` en local y dentro de Docker (donde se
conserva la misma estructura de carpetas).

| Variable              | Uso                                   | Por defecto              |
|-----------------------|---------------------------------------|--------------------------|
| MODELOS_DIR           | Carpeta con las versiones del modelo  | fase-2/models            |
| MODELO_VERSION        | Versión que usa /predict              | v1                       |
| DATOS_ENTRENAMIENTO   | CSV que usa /train                    | fase-2/data/datos.csv    |
| TARGET                | Variable objetivo                     | Transported              |
| SEMILLA               | Semilla de /train                     | 42                       |
| TEST_SIZE             | Proporción de grupos de prueba        | 0.2                      |
| CV_FOLDS              | Folds de la validación cruzada        | 5 (0 la omite)           |

Una ruta relativa en una variable de entorno se interpreta desde la carpeta en
la que se inicia el servidor.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DIR_FASE3 = Path(__file__).resolve().parent
DIR_FASE2 = DIR_FASE3.parent / "fase-2"


@dataclass(frozen=True)
class Configuracion:
    modelos_dir: Path
    modelo_version: str
    datos_entrenamiento: Path
    target: str = "Transported"
    semilla: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    @classmethod
    def desde_entorno(cls, entorno: Mapping[str, str] | None = None) -> "Configuracion":
        """Lee la configuración de `entorno` (por defecto, `os.environ`).

        Lanza ValueError con el nombre de la variable si un valor no es válido.
        """
        entorno = os.environ if entorno is None else entorno

        def texto(nombre: str, defecto: str) -> str:
            valor = entorno.get(nombre, "").strip()
            return valor or defecto

        def numero(nombre: str, tipo: type, defecto):
            valor = entorno.get(nombre, "").strip()
            if not valor:
                return defecto
            try:
                return tipo(valor)
            except ValueError as error:
                raise ValueError(
                    f"La variable de entorno {nombre}={valor!r} no es un "
                    f"{'entero' if tipo is int else 'número'} válido."
                ) from error

        return cls(
            modelos_dir=Path(texto("MODELOS_DIR", str(DIR_FASE2 / "models"))),
            modelo_version=texto("MODELO_VERSION", "v1"),
            datos_entrenamiento=Path(
                texto("DATOS_ENTRENAMIENTO", str(DIR_FASE2 / "data" / "datos.csv"))
            ),
            target=texto("TARGET", "Transported"),
            semilla=numero("SEMILLA", int, 42),
            test_size=numero("TEST_SIZE", float, 0.2),
            cv_folds=numero("CV_FOLDS", int, 5),
        )
