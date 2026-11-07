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
- La lista de siete variables elegida en la Fase 1 se verificó de nuevo usando
  solo la partición de entrenamiento y cada fold de la validación cruzada (ver
  [Verificación de la selección de variables](#verificación-de-la-selección-de-variables)).
  En esta fase la lista queda fija en el código y no se vuelve a seleccionar
  con los datos.
- Se agregan validación de entradas, almacenamiento versionado, pruebas
  automatizadas y ejecución con Docker.

Las demás correcciones pedidas en la retroalimentación de la Fase 1 se
aplicaron en `fase-1/` y están documentadas en su
[README](../fase-1/README.md) y en el notebook:

- Los porcentajes de nulos se reportan sin redondear (entre 2,059% y
  2,496%). En la Fase 1 el dataset se propuso y se aprobó con el 2% que
  muestra la página de Kaggle para cada columna, (seguramente con un 
  valor redondeado), y (error nuestro) no se contrastó después con el 
  cálculo exacto sobre los datos. Esos porcentajes superan ligeramente el 
  máximo de la guía (0,1%–2%); se reconoce el incumplimiento y no se 
  redondean las cifras para afirmar lo contrario.
- Se distingue el modelo evaluado al inicio (accuracy 0,8010) del pipeline
  guardado en la Entrega 1 (0,8005) y del pipeline corregido con mediana
  (0,7993).
- El notebook lee el CSV local o, si no existe, un commit fijo, y las versiones
  de las librerías quedan documentadas en `fase-1/requirements.txt`.
- Se moderaron las conclusiones del EDA: un p-valor no significativo no
  demuestra ausencia de relación, y un gasto cero no confirma por sí solo la
  edad registrada.

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

Ambas versiones superan con claridad al modelo base (`DummyClassifier`, que
predice la clase más frecuente del entrenamiento): unos 28–30 puntos de
accuracy en prueba.

Cada corrida sobre el ejemplo generó **14 predicciones**. Los resultados se
guardan en `outputs/predicciones_v1.csv` y `outputs/predicciones_v2.csv`
(archivos ignorados por Git para no mezclar ejecuciones locales).

Salida real de `predict.py` con v1 y primeras filas del resultado:

```text
Predicciones generadas: 14
Archivo guardado en: outputs\predicciones_v1.csv

PassengerId,prediccion,probabilidad
0001_01,1,0.5766031273183951
0002_01,0,0.3974458651492998
0003_01,0,0.09741403224666194
```

Intentar crear de nuevo una versión existente falla sin modificarla (código de
salida 2):

```text
train.py: error: La versión 'v1' ya existe en models\v1; use otro --version.
```

## Pruebas

Desde `fase-2`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Las pruebas cubren entrenamiento y artefactos, predicción antes y después de
guardar, valores faltantes, columnas ausentes, versionamiento, tipos inválidos,
modelo inexistente y ayuda de los scripts. Usan datos sintéticos y directorios
temporales, sin descargas de red. Resultado obtenido:

```text
.............                                                            [100%]
13 passed in 16.94s
```

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

Salida real de los dos comandos anteriores (el entrenamiento usa
`--cv-folds 0`, por eso no reporta CV):

```text
Accuracy Random Forest (test): 0.7811
Accuracy modelo base (test): 0.4816
Versión guardada en: models/v-docker
Archivos guardados: modelo.joblib, metrics.json, metadata.json

Predicciones generadas: 14
Archivo guardado en: outputs/predicciones_docker.csv
```

La imagen usa `python:3.13-slim` y no entrena durante `docker build`.
Los montajes hacen que los modelos y predicciones permanezcan en el equipo
anfitrión después de que `--rm` elimine los contenedores. La versión
`v-docker` es solo de verificación y no se versiona en Git.

Si se usa Git Bash en Windows en lugar de PowerShell, hay que anteponer
`MSYS_NO_PATHCONV=1` y usar `$(pwd -W)` en las rutas de `-v`; si no, Git Bash
reescribe las rutas `/app/...`.

## Verificación de la selección de variables

La retroalimentación de la Fase 1 señaló que las siete variables se eligieron
con asociaciones calculadas antes de separar los datos. Para comprobar si la
lista depende de los datos de prueba, `seleccion_variables.py` recalcula los
criterios de la Fase 1 (|r| de Pearson ≥ 0,2 para numéricas y binarias; V de
Cramér ≥ 0,2 para categóricas) usando solo la partición de entrenamiento y cada
fold, con la misma partición que `train.py`:

```powershell
.\.venv\Scripts\python.exe seleccion_variables.py --data data/datos.csv --seed 42 --output outputs/seleccion_variables_s42.csv
.\.venv\Scripts\python.exe seleccion_variables.py --data data/datos.csv --seed 123 --output outputs/seleccion_variables_s123.csv
```

Resumen en las 6 particiones evaluadas (entrenamiento completo y 5 folds):

| Variable | Medida | Semilla 42 (v1) | Supera 0,2 | Semilla 123 (v2) | Supera 0,2 |
|---|---|---:|---:|---:|---:|
| `CryoSleep` | r | 0,456–0,471 | todas | 0,465–0,472 | todas |
| `RoomService` | r | 0,241–0,256 | todas | 0,245–0,254 | todas |
| `Spa` | r | 0,215–0,223 | todas | 0,216–0,224 | todas |
| `Deck` | V | 0,194–0,216 | 3 de 6 | 0,215–0,226 | todas |
| `VRDeck` | r | 0,196–0,208 | 4 de 6 | 0,205–0,210 | todas |
| `HomePlanet` | V | 0,184–0,209 | 1 de 6 | 0,196–0,204 | 1 de 6 |
| `Age` | r | 0,075–0,085 | ninguna | 0,065–0,077 | ninguna |
| Mejor descartada (`Group_Size`) | V | 0,119–0,131 | ninguna | 0,122–0,135 | ninguna |

Valores en valor absoluto. `HomePlanet` y `Age` ya se conservaban en la Fase 1
por criterio explícito, no por el umbral.

Lectura: ninguna variable descartada alcanza el umbral en ninguna partición, y
la separación con las elegidas es amplia. Pero el umbral de 0,2 es frágil en el
límite: aplicado de forma estricta en cada fold, `Deck` y `VRDeck` entrarían o
saldrían según la partición. Por eso no se puede afirmar que la fuga de la
Fase 1 fue "leve" en cuanto a la métrica; sí que **la lista de variables no
depende de los datos de prueba**. Se mantiene fija, en lugar de seleccionar
dentro del pipeline, para que todas las versiones tengan el mismo esquema de
entrada en `predict.py` y en la API de la siguiente fase.

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
- La accuracy de esta fase (≈0,781) es menor que la del pipeline corregido de
  la Fase 1 (0,7993) por hiperparámetros fijos sin búsqueda, otra partición
  por grupos y la imputación determinista por mediana.

### Fuentes y crédito

- Datos: [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic).
- Documentación consultada: [scikit-learn](https://scikit-learn.org/stable/)
  (`Pipeline`, `ColumnTransformer`, `GroupShuffleSplit`,
  `StratifiedGroupKFold`, `DummyClassifier`), [pandas](https://pandas.pydata.org/docs/),
  [SciPy](https://docs.scipy.org/doc/scipy/) (`scipy.stats.contingency.association`
  para la V de Cramér), [pytest](https://docs.pytest.org/) y
  [Docker](https://docs.docker.com/).
- Código externo: no se copió ni adaptó código de terceros. Los módulos de
  `src/` reorganizan el código propio de la [Fase 1](../fase-1/README.md).
La versión evaluada de esta entrega se identifica con la etiqueta Git
`entrega-2`.
