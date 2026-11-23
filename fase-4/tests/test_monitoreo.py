"""Pruebas del registro y monitoreo (Fase 4).

Cada prueba entrena un modelo sintético en ``tmp_path`` (igual que las
pruebas de la Fase 3) y usa el ``TestClient`` de FastAPI sin red.
"""

from dataclasses import replace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import crear_app
from configuracion import Configuracion
from registro import registrar, ENCABEZADO
from monitoreo import leer_registro, calcular_indicadores
from src.almacenamiento import guardar_version
from src.modelo import crear_modelo
from src.preprocesamiento import FEATURES, preparar_datos


@pytest.fixture
def api_sintetica(tmp_path):
    """Crea un modelo sintético y la configuración para las pruebas."""
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
def ruta_registro(tmp_path):
    return tmp_path / "logs" / "predicciones.csv"


def _crear_cliente_con_registro(config, ruta_registro):
    """Crea un TestClient con el observador de registro conectado."""
    from registro import registrar as _registrar

    def observador(estado, version, prediccion):
        _registrar(ruta_registro, estado, version, prediccion)

    return TestClient(crear_app(config, observador=observador))


def test_registro_dos_validas_una_invalida(api_sintetica, ruta_registro):
    """2 válidas + 1 inválida → 3 filas y 33.3 % de fallos."""
    config, pasajero = api_sintetica
    cliente = _crear_cliente_con_registro(config, ruta_registro)

    cliente.post("/predict", json=pasajero)
    cliente.post("/predict", json=pasajero)
    sin_age = {k: v for k, v in pasajero.items() if k != "Age"}
    cliente.post("/predict", json=sin_age)

    filas = leer_registro(ruta_registro)
    assert len(filas) == 3
    total, porcentaje = calcular_indicadores(filas)
    assert total == 3
    assert porcentaje == 33.3


def test_registro_vacio_sin_solicitudes(ruta_registro):
    """Registro vacío o inexistente → total 0 y sin solicitudes."""
    filas = leer_registro(ruta_registro)
    total, porcentaje = calcular_indicadores(filas)
    assert total == 0
    assert porcentaje is None


def test_rechazo_genera_una_sola_fila(api_sintetica, ruta_registro):
    """Un rechazo de validación genera exactamente una fila."""
    config, pasajero = api_sintetica
    cliente = _crear_cliente_con_registro(config, ruta_registro)

    sin_age = {k: v for k, v in pasajero.items() if k != "Age"}
    respuesta = cliente.post("/predict", json=sin_age)
    assert respuesta.status_code == 422

    filas = leer_registro(ruta_registro)
    assert len(filas) == 1
    assert filas[0]["estado_http"] == "422"


def test_registro_503_modelo_inexistente(api_sintetica, ruta_registro):
    """Modelo inexistente → 503 con una fila en el registro."""
    config, pasajero = api_sintetica
    config_v99 = replace(config, modelo_version="v99")
    cliente = _crear_cliente_con_registro(config_v99, ruta_registro)

    respuesta = cliente.post("/predict", json=pasajero)
    assert respuesta.status_code == 503

    filas = leer_registro(ruta_registro)
    assert len(filas) == 1
    assert filas[0]["estado_http"] == "503"
    assert filas[0]["prediccion"] == ""


def test_registro_persiste_entre_ejecuciones(api_sintetica, ruta_registro):
    """El registro conserva filas entre reinicios de la API."""
    config, pasajero = api_sintetica

    # Primera "ejecución"
    cliente1 = _crear_cliente_con_registro(config, ruta_registro)
    cliente1.post("/predict", json=pasajero)

    # Segunda "ejecución" (nuevo TestClient, mismo archivo)
    cliente2 = _crear_cliente_con_registro(config, ruta_registro)
    cliente2.post("/predict", json=pasajero)

    filas = leer_registro(ruta_registro)
    assert len(filas) == 2
