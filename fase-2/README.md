# Fase 2: scripts y Docker del Titanic Espacial

## Equipo y propósito

**Modelos y Simulación de Sistemas I — Proyecto Spaceship Titanic**

Integrantes: Miguel Angel Avendaño Parra, Miguel Andres Mejía Cuadrado y
Samuel Tabares García.

La aplicación predice si un pasajero del dataset Spaceship Titanic fue
transportado. Esta fase convierte el flujo de la [Fase 1](../fase-1/README.md)
en scripts reproducibles de entrenamiento y predicción, conserva dos versiones
del modelo y permite ejecutar ambos procesos localmente o mediante Docker.

## Cambios respecto a la Fase 1

- La imputación, la codificación y las transformaciones que aprenden de los
  datos están dentro de un `Pipeline` ajustado solo con entrenamiento.
- `Age` se imputa con la mediana aprendida, no con muestreo aleatorio.
- La separación conserva los grupos naturales de viaje mediante
  `GroupShuffleSplit`, y la validación cruzada usa
  `StratifiedGroupKFold`.
- Se agregan validación de entradas, almacenamiento versionado, pruebas
  automatizadas y ejecución con Docker.

## Entorno e instalación

Se usa **Python 3.13** tanto localmente como en Docker. Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Las versiones de las dependencias están fijadas en
[requirements.txt](requirements.txt). Para una copia nueva, el dataset
completo se toma de `../Data_Set.csv` y se copia a `data/datos.csv`:

```powershell
Copy-Item ..\Data_Set.csv data\datos.csv
```

El CSV se mantiene en la raíz porque la Fase 1 lo utiliza desde esa ubicación;
`data/datos.csv` está ignorado por Git. El SHA-256 de los datos usados en las
versiones entregadas es
`92515331df73abce3b7c01dba35c33971027268bd84609bc6593ca43b18e649f`.

## Datos de predicción

El archivo [data/ejemplo_prediccion.csv](data/ejemplo_prediccion.csv) no
contiene `Transported`. Incluye `PassengerId`, `HomePlanet`, `CryoSleep`,
`Cabin`, `Destination`, `Age`, `VIP`, `RoomService`, `FoodCourt`,
`ShoppingMall`, `Spa`, `VRDeck` y `Name`.

El modelo usa estas siete variables, en este orden:
`HomePlanet`, `CryoSleep`, `Deck`, `Age`, `RoomService`, `Spa` y `VRDeck`.
`Deck` se obtiene de `Cabin` cuando no viene como columna. Las columnas
adicionales se ignoran, los valores faltantes se imputan con las reglas
guardadas en el pipeline y las categorías no observadas se aceptan mediante
`handle_unknown="ignore"`.

## Entrenamiento y predicción

Consultar las opciones:

```powershell
.\.venv\Scripts\python.exe train.py --help
.\.venv\Scripts\python.exe predict.py --help
```

Regenerar las versiones entregadas:

```powershell
.\.venv\Scripts\python.exe train.py --data data/datos.csv --target Transported --output-dir models --version v1 --seed 42 --test-size 0.2 --cv-folds 5
.\.venv\Scripts\python.exe train.py --data data/datos.csv --target Transported --output-dir models --version v2 --seed 123 --test-size 0.2 --cv-folds 5
```

Generar predicciones:

```powershell
.\.venv\Scripts\python.exe predict.py --model models\v1\modelo.joblib --input data\ejemplo_prediccion.csv --output outputs\predicciones_v1.csv
.\.venv\Scripts\python.exe predict.py --model models\v2\modelo.joblib --input data\ejemplo_prediccion.csv --output outputs\predicciones_v2.csv
```

Cada versión contiene `modelo.joblib`, `metrics.json` y `metadata.json` en
`models/v1/` o `models/v2/`. La salida contiene `PassengerId`, `prediccion`
(`1` = transportado, `0` = no transportado) y `probabilidad`, cuando el
modelo puede calcularla. El orden de las filas de entrada se conserva.

## Evidencia de las versiones

Las métricas se calcularon sobre la misma separación por grupos y con
`accuracy`:

| Versión | Semilla | Accuracy prueba | Modelo base | CV media ± desviación |
|---|---:|---:|---:|---:|
| v1 | 42 | 0.7811 | 0.4816 | 0.7898 ± 0.0089 |
| v2 | 123 | 0.7812 | 0.5052 | 0.7867 ± 0.0068 |

Cada corrida sobre el ejemplo generó **14 predicciones**. Los resultados se
guardan en `outputs/predicciones_v1.csv` y `outputs/predicciones_v2.csv`
(archivos ignorados por Git para no mezclar ejecuciones locales).

## Pruebas

Desde `fase-2`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Las pruebas cubren entrenamiento y artefactos, predicción antes y después de
guardar, valores faltantes, columnas ausentes, versionamiento, tipos inválidos,
modelo inexistente y ayuda de los scripts. Usan datos sintéticos y directorios
temporales, sin descargas de red.

## Docker

Construir la imagen desde `fase-2`:

```powershell
docker build -t modelos-fase2:1.0 .
```

Entrenar montando los datos en solo lectura y persistiendo `models` en el
equipo anfitrión:

```powershell
$commit = git -C .. rev-parse HEAD
docker run --rm -e GIT_COMMIT=$commit -v "${PWD}/../Data_Set.csv:/app/data/datos.csv:ro" -v "${PWD}/models:/app/models" modelos-fase2:1.0 python train.py --data data/datos.csv --target Transported --output-dir models --version v-docker --seed 42 --test-size 0.2 --cv-folds 0
```

Predecir usando el modelo persistido:

```powershell
docker run --rm -v "${PWD}/models:/app/models:ro" -v "${PWD}/data:/app/data:ro" -v "${PWD}/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```

La imagen usa `python:3.13-slim` y no entrena durante `docker build`.
Los montajes hacen que los modelos y predicciones permanezcan en el equipo
anfitrión después de que `--rm` elimine los contenedores.

## Estructura mínima

`train.py` y `predict.py` son los puntos de entrada. La carpeta `src/`
contiene la lógica reutilizable de datos, preprocesamiento, partición,
modelado y almacenamiento, cumpliendo la separación indicada en la
especificación. `seleccion_variables.py` es un script auxiliar reproducible
para verificar la decisión de variables; no es un punto de entrada de la
aplicación.

## Limitaciones y fuentes

- La lista fija de siete variables se conserva para mantener un esquema
  estable entre versiones y las futuras fases. El umbral de asociación de
  `0.2` es frágil para `Deck` y `VRDeck`; la selección se verificó solo con
  particiones de entrenamiento y no con datos de prueba.
- Random Forest puede producir pequeñas diferencias si se cambian semillas,
  versiones o parámetros.
- Las categorías nuevas se ignoran en el one-hot; sus columnas codificadas
  quedan en cero.
- La diferencia frente a la métrica de Fase 1 se debe a hiperparámetros fijos,
  otra partición y la imputación determinista por mediana.

El dataset proviene de
[Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic).
La versión evaluada de esta entrega se identifica con la etiqueta Git
`entrega-2`.
