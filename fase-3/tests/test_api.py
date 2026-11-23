from dataclasses import replace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import crear_app
from configuracion import Configuracion
from src.almacenamiento import cargar_version, guardar_version
from src.modelo import crear_modelo
from src.preprocesamiento import FEATURES, preparar_datos


@pytest.fixture
def api_sintetica(tmp_path):
    filas = []
    for indice in range(40):
        grupo = 1000 + indice
        transportado = indice % 2 == 0
        filas.append(
            {
                "PassengerId": f"{grupo:04d}_01",
                "HomePlanet": ("Europa", "Earth", "Mars")[indice % 3],
                "CryoSleep": indice % 4 == 0,
                "Cabin": f"{('A', 'B', 'F', 'G')[indice % 4]}/{indice}/P",
                "Age": 18 + (indice * 3) % 55,
                "RoomService": 0 if transportado else 20 + indice,
                "Spa": 10 + (indice * 2) % 40 if transportado else 0,
                "VRDeck": 5 + indice if transportado else 0,
                "Transported": transportado,
            }
        )

    datos = pd.DataFrame(filas)
    ruta_datos = tmp_path / "datos_sinteticos.csv"
    datos.to_csv(ruta_datos, index=False)

    modelo = crear_modelo(semilla=17)
    modelo.named_steps["rf"].set_params(n_estimators=20, n_jobs=1)
    modelo.fit(preparar_datos(datos), datos["Transported"].astype(int))

    modelos_dir = tmp_path / "models"
    guardar_version(
        modelo,
        {"accuracy_test": 1.0},
        {"version": "v1", "features": list(FEATURES)},
        modelos_dir,
        "v1",
    )
    config = Configuracion(
        modelos_dir=modelos_dir,
        modelo_version="v1",
        datos_entrenamiento=ruta_datos,
        semilla=17,
        test_size=0.2,
        cv_folds=0,
    )
    pasajero = datos.iloc[0].drop(labels="Transported").to_dict()
    return config, pasajero


@pytest.fixture
def cliente(api_sintetica):
    config, _ = api_sintetica
    with TestClient(crear_app(config)) as test_client:
        yield test_client


def test_health_responde_ok(cliente):
    respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "ok"


def test_predict_entrada_valida(cliente, api_sintetica):
    config, pasajero = api_sintetica

    respuesta = cliente.post("/predict", json=pasajero)

    assert respuesta.status_code == 200
    assert {"prediccion", "version"} <= respuesta.json().keys()
    assert respuesta.json()["version"] == config.modelo_version


def test_predict_campo_obligatorio_ausente(cliente, api_sintetica):
    _, pasajero = api_sintetica
    pasajero.pop("Age")

    respuesta = cliente.post("/predict", json=pasajero)

    assert respuesta.status_code == 422
    assert "Age" in respuesta.json()["detail"]


def test_predict_coincide_con_predict_py(cliente, api_sintetica):
    from predict import _generar_predicciones

    config, pasajero = api_sintetica
    modelo, _, _ = cargar_version(config.modelos_dir / config.modelo_version)
    esperado = _generar_predicciones(modelo, pd.DataFrame([pasajero])).iloc[0]

    respuesta = cliente.post("/predict", json=pasajero)

    assert respuesta.status_code == 200
    assert respuesta.json()["prediccion"] is bool(int(esperado["prediccion"]))
    assert respuesta.json()["probabilidad"] == pytest.approx(
        float(esperado["probabilidad"]), abs=1e-12
    )


def test_predict_nulo_permitido(cliente, api_sintetica):
    _, pasajero = api_sintetica
    pasajero["Age"] = None

    respuesta = cliente.post("/predict", json=pasajero)

    assert respuesta.status_code == 200


def test_predict_tipo_incompatible(cliente, api_sintetica):
    _, pasajero = api_sintetica
    pasajero["Age"] = "abc"

    respuesta = cliente.post("/predict", json=pasajero)

    assert respuesta.status_code == 422
    assert "Age" in respuesta.json()["detail"]


def test_predict_modelo_inexistente(api_sintetica):
    config, pasajero = api_sintetica
    with TestClient(crear_app(replace(config, modelo_version="v99"))) as cliente_v99:
        respuesta = cliente_v99.post("/predict", json=pasajero)

    assert respuesta.status_code == 503
    assert "MODELO_VERSION" in respuesta.json()["detail"]


def test_train_crea_version_sin_tocar_anteriores(api_sintetica):
    config, _ = api_sintetica
    ruta_v1 = config.modelos_dir / "v1"
    archivos_v1 = {
        nombre: (ruta_v1 / nombre).read_bytes()
        for nombre in ("modelo.joblib", "metrics.json", "metadata.json")
    }

    with TestClient(crear_app(config)) as cliente:
        respuesta = cliente.post("/train")

    assert respuesta.status_code == 201
    assert respuesta.json()["version"] == "v2"
    assert respuesta.json()["version_activa"] == "v1"
    assert (config.modelos_dir / "v2").is_dir()
    assert {
        nombre: (ruta_v1 / nombre).read_bytes()
        for nombre in archivos_v1
    } == archivos_v1
    _, _, metadata_v1 = cargar_version(ruta_v1)
    assert metadata_v1["version"] == "v1"
