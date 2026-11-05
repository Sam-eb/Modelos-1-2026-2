"""Pruebas de la aplicación de la Fase 2 (punto 12 de la especificación).

Las cinco pruebas mínimas son:
1. Entrenamiento y almacenamiento.
2. Carga y predicción.
3. Valores faltantes.
4. Entrada inválida.
5. Versionamiento.
Al final hay pruebas adicionales de errores previsibles y de `--help`.

Ejecutar desde `fase-2` con: python -m pytest
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import predict
import train
from conftest import entrenar_version
from src.almacenamiento import cargar_version, guardar_version
from src.modelo import crear_modelo
from src.preprocesamiento import FEATURES, preparar_datos


def predecir(modelo: Path, entrada: Path, salida: Path) -> int:
    """Ejecuta predict.py como desde la línea de comandos."""
    return predict.main(
        ["--model", str(modelo), "--input", str(entrada), "--output", str(salida)]
    )


# 1. Entrenamiento y almacenamiento ------------------------------------------

def test_entrenamiento_genera_artefactos(version_entrenada, csv_entrenamiento, datos_entrenamiento):
    for archivo in ("modelo.joblib", "metrics.json", "metadata.json"):
        assert (version_entrenada / archivo).is_file()

    modelo, metricas, metadata = cargar_version(version_entrenada)

    # El artefacto es el pipeline completo (preprocesamiento + modelo).
    assert list(modelo.named_steps) == ["preprocesamiento", "rf"]
    assert hasattr(modelo, "predict")

    # Métricas del modelo y del modelo base sobre la misma partición.
    assert 0.0 <= metricas["accuracy_test"] <= 1.0
    assert 0.0 <= metricas["accuracy_test_base"] <= 1.0
    assert metricas["accuracy_cv_mean"] is None  # --cv-folds 0

    # Metadatos mínimos (punto 10.1 de la especificación).
    assert metadata["version"] == "v1"
    assert metadata["target"] == "Transported"
    assert metadata["features"] == FEATURES
    assert metadata["semilla"] == 0
    sha = hashlib.sha256(csv_entrenamiento.read_bytes()).hexdigest()
    assert metadata["datos"]["sha256"] == sha
    obs = metadata["observaciones"]
    assert obs["total"] == len(datos_entrenamiento)
    assert obs["entrenamiento"] + obs["prueba"] == obs["total"]
    assert metadata["version_python"]
    assert "scikit-learn" in metadata["librerias"]


# 2. Carga y predicción -------------------------------------------------------

def test_prediccion_coincide_con_modelo_antes_de_guardar(
    tmp_path, datos_entrenamiento, datos_prediccion
):
    # Modelo ajustado en memoria, antes de guardarlo.
    X = preparar_datos(datos_entrenamiento)
    y = datos_entrenamiento["Transported"].astype(int)
    modelo = crear_modelo(semilla=0).fit(X, y)
    esperado = modelo.predict(preparar_datos(datos_prediccion))
    proba_esperada = modelo.predict_proba(preparar_datos(datos_prediccion))[:, 1]

    metadata = {"version": "memoria", "features": FEATURES}
    directorio = guardar_version(modelo, {}, metadata, tmp_path / "models", "memoria")

    entrada = tmp_path / "entrada.csv"
    salida = tmp_path / "predicciones.csv"
    datos_prediccion.to_csv(entrada, index=False)
    assert predecir(directorio, entrada, salida) == 0

    resultado = pd.read_csv(salida)
    # Una predicción por fila, en el mismo orden de la entrada.
    assert len(resultado) == len(datos_prediccion)
    assert resultado["PassengerId"].tolist() == datos_prediccion["PassengerId"].tolist()
    # El artefacto guardado reproduce exactamente al modelo en memoria.
    np.testing.assert_array_equal(resultado["prediccion"].to_numpy(), esperado)
    np.testing.assert_allclose(resultado["probabilidad"].to_numpy(), proba_esperada)


# 3. Valores faltantes --------------------------------------------------------

def test_valores_faltantes_se_procesan(tmp_path, version_entrenada, datos_prediccion):
    datos = datos_prediccion.copy()
    datos["CryoSleep"] = datos["CryoSleep"].astype(object)  # admite NaN
    datos.loc[0, "Age"] = np.nan
    datos.loc[1, "CryoSleep"] = np.nan
    datos.loc[2, "HomePlanet"] = np.nan
    datos.loc[3, "Cabin"] = np.nan
    datos.loc[4, ["RoomService", "Spa", "VRDeck"]] = np.nan

    entrada = tmp_path / "con_nulos.csv"
    salida = tmp_path / "predicciones.csv"
    datos.to_csv(entrada, index=False)
    assert predecir(version_entrenada, entrada, salida) == 0

    resultado = pd.read_csv(salida)
    # No se eliminan filas y todas reciben predicción válida.
    assert len(resultado) == len(datos)
    assert resultado["prediccion"].isin([0, 1]).all()
    assert resultado["probabilidad"].between(0, 1).all()

    # Age faltante se reemplaza por la mediana aprendida en entrenamiento.
    modelo, _, _ = cargar_version(version_entrenada)
    imputador = modelo.named_steps["preprocesamiento"].named_steps["imputar"]
    imputado = imputador.transform(preparar_datos(datos.iloc[[0]]))
    assert imputado["Age"].iloc[0] == imputador.mediana_age_

    # La imputación es determinista: la fila con Age nula da el mismo
    # resultado sola que dentro del lote (problema señalado en la Fase 1).
    sola = tmp_path / "sola.csv"
    salida_sola = tmp_path / "prediccion_sola.csv"
    datos.iloc[[0]].to_csv(sola, index=False)
    assert predecir(version_entrenada, sola, salida_sola) == 0
    assert pd.read_csv(salida_sola)["probabilidad"].iloc[0] == pytest.approx(
        resultado["probabilidad"].iloc[0]
    )


def test_cabin_completamente_vacia_se_procesa(tmp_path, version_entrenada, datos_prediccion):
    # Con una sola fila y Cabin nula, pandas lee la columna como numérica.
    datos = datos_prediccion.iloc[[0]].copy()
    datos["Cabin"] = np.nan
    entrada = tmp_path / "cabin_vacia.csv"
    salida = tmp_path / "predicciones.csv"
    datos.to_csv(entrada, index=False)

    assert predecir(version_entrenada, entrada, salida) == 0
    assert len(pd.read_csv(salida)) == 1


# 4. Entrada inválida ---------------------------------------------------------

def test_columna_requerida_ausente_falla(tmp_path, version_entrenada, datos_prediccion, capsys):
    entrada = tmp_path / "sin_age.csv"
    salida = tmp_path / "predicciones.csv"
    datos_prediccion.drop(columns="Age").to_csv(entrada, index=False)

    with pytest.raises(SystemExit) as error:
        predecir(version_entrenada, entrada, salida)

    assert error.value.code != 0
    assert "Age" in capsys.readouterr().err
    # No queda un archivo que aparente contener resultados válidos.
    assert not salida.exists()


# 5. Versionamiento -----------------------------------------------------------

def test_version_no_sobrescribe(tmp_path, csv_entrenamiento, datos_prediccion):
    models = tmp_path / "models"
    assert entrenar_version(csv_entrenamiento, models, "v1", semilla=0) == 0
    assert entrenar_version(csv_entrenamiento, models, "v2", semilla=1) == 0

    # Ambas versiones existen, conservan sus metadatos y pueden cargarse.
    for version, semilla in (("v1", 0), ("v2", 1)):
        _, _, metadata = cargar_version(models / version)
        assert metadata["version"] == version
        assert metadata["semilla"] == semilla

    # Un identificador repetido falla sin modificar la versión existente.
    antes = {p.name: p.read_bytes() for p in (models / "v1").iterdir()}
    with pytest.raises(SystemExit) as error:
        entrenar_version(csv_entrenamiento, models, "v1", semilla=5)
    assert error.value.code != 0
    despues = {p.name: p.read_bytes() for p in (models / "v1").iterdir()}
    assert despues == antes

    # predict.py permite elegir explícitamente cualquiera de las dos.
    entrada = tmp_path / "entrada.csv"
    datos_prediccion.to_csv(entrada, index=False)
    for version in ("v1", "v2"):
        salida = tmp_path / f"pred_{version}.csv"
        assert predecir(models / version, entrada, salida) == 0
        assert len(pd.read_csv(salida)) == len(datos_prediccion)


# Pruebas adicionales: errores previsibles y ayuda -----------------------------

@pytest.mark.parametrize(
    ("modificar", "mensaje"),
    [
        (lambda d: d.assign(Spa=-5.0), "negativos"),
        (lambda d: d.assign(Age="desconocida"), "no numéricos"),
        (lambda d: d.assign(CryoSleep="tal vez"), "booleano"),
    ],
    ids=["gasto_negativo", "edad_no_numerica", "cryosleep_invalido"],
)
def test_tipo_incompatible_falla(
    tmp_path, version_entrenada, datos_prediccion, capsys, modificar, mensaje
):
    entrada = tmp_path / "invalido.csv"
    salida = tmp_path / "predicciones.csv"
    modificar(datos_prediccion.copy()).to_csv(entrada, index=False)

    with pytest.raises(SystemExit) as error:
        predecir(version_entrenada, entrada, salida)

    assert error.value.code != 0
    assert mensaje in capsys.readouterr().err
    assert not salida.exists()


def test_modelo_inexistente_falla(tmp_path, datos_prediccion, capsys):
    entrada = tmp_path / "entrada.csv"
    salida = tmp_path / "predicciones.csv"
    datos_prediccion.to_csv(entrada, index=False)

    with pytest.raises(SystemExit) as error:
        predecir(tmp_path / "models" / "no_existe", entrada, salida)

    assert error.value.code != 0
    assert "no existe" in capsys.readouterr().err
    assert not salida.exists()


def test_predict_acepta_ruta_al_joblib(tmp_path, version_entrenada, datos_prediccion):
    entrada = tmp_path / "entrada.csv"
    datos_prediccion.to_csv(entrada, index=False)
    por_carpeta = tmp_path / "carpeta.csv"
    por_archivo = tmp_path / "archivo.csv"

    assert predecir(version_entrenada, entrada, por_carpeta) == 0
    assert predecir(version_entrenada / "modelo.joblib", entrada, por_archivo) == 0
    pd.testing.assert_frame_equal(pd.read_csv(por_carpeta), pd.read_csv(por_archivo))


@pytest.mark.parametrize("script", [train, predict], ids=["train", "predict"])
def test_help_disponible(script, capsys):
    with pytest.raises(SystemExit) as error:
        script.main(["--help"])
    assert error.value.code == 0
    assert "usage" in capsys.readouterr().out
