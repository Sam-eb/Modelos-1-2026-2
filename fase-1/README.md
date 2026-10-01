# El extraño caso del Titanic Espacial

## Integrantes del equipo
- Miguel Angel Avendaño Parra
- Miguel Andres Mejía Cuadrado
- Samuel Tabares García 

## Descripción del problema
Es el año 2912. En esta época de avances tecnológicos los viajes espaciales son cosa del día a día. Pero algo raro sucede en la terminal interplanetaria de La Tierra; acaba de llegar una transmisión de una nave a 4 años luz, y al parecer tiene problemas.

El Titanic Espacial es una nueva línea de naves interestelares, tiene una capacidad de 13.000 pasajeros, gracias a esta capacidad se utiliza para transportar personas desde nuestro sistema solar hacia 3 exoplanetas habitables que orbitan alrededor de una estrella.

Durante el viaje, el Titanic Espacial colisionó con una anomalía espacio-temporal que se escondía en una nube de polvo espacial. Al igual que su antepasado, el Titanic Espacial fue escenario de una gran tragedia; debido a la colisión casi la mitad de los pasajeros fueron enviados a otra dimensión. 

Nuestra misión es utilizar herramientas de predicción para ayudar a determinar qué pasajeros fueron enviados a esta nueva dimensión mediante los registros recuperados de los sistemas de la nave.

## Fuente del conjunto de datos

El conjunto de datos corresponde a la competencia **Spaceship Titanic** de Kaggle:

https://www.kaggle.com/competitions/spaceship-titanic

El archivo utilizado por el notebook es `Data_Set.csv`, ubicado en la raíz del repositorio.

## Estructura de la entrega

```
Data_Set.csv
README.md
fase-1/
├── notebook.ipynb
├── preprocesamiento.py
└── modelo.joblib
```

## Objetivo del modelo

Clasificar, según los datos registrados del pasajero, si fue transportado a otra dimensión mediante la variable objetivo `Transported`.

Antes del entrenamiento se realiza una preparación que incluye:

- separación de entrenamiento y prueba mediante grupos de viaje;
- imputación de valores faltantes;
- selección de las variables definidas durante la exploración;
- codificación one-hot de las variables categóricas.

La separación y la validación usan `StratifiedGroupKFold` con 5 folds, evitando que pasajeros del mismo grupo aparezcan simultáneamente en entrenamiento y validación. El conjunto de prueba se mantiene reservado para la evaluación final del modelo seleccionado.

## Algoritmos utilizados

Se compararon los siguientes candidatos:

- modelo base (`DummyClassifier`);
- regresión logística, con `StandardScaler` para las variables numéricas;
- árbol de decisión con `max_depth=7`;
- Random Forest, con una búsqueda inicial de hiperparámetros;
- Gradient Boosting, con `n_estimators=100`, `learning_rate=0.05` y `max_depth=3`;
- SVM con kernel RBF, usando `StandardScaler` y una búsqueda inicial de `C` y `gamma`.

Los modelos basados en árboles no requieren escalamiento porque toman decisiones mediante umbrales, mientras que la regresión logística y el SVM sí lo incorporan dentro de un pipeline para evitar fuga de información.

El modelo seleccionado y almacenado es **Random Forest**.

## Métrica empleada

La métrica principal es **accuracy**, es decir, la proporción de predicciones correctas. Es adecuada porque las clases de `Transported` están prácticamente balanceadas: aproximadamente 49,6 % son `False` y 50,4 % son `True`. Como apoyo, también se revisa el `classification_report`, que incluye precision, recall y F1-score por clase.

## Principales resultados obtenidos

Los siguientes resultados corresponden a la accuracy media y desviación estándar obtenidas mediante validación cruzada estratificada y agrupada sobre `X_train`:

| Modelo | Accuracy CV |
| --- | ---: |
| Modelo base | 0.5036 ± 0.0002 |
| Regresión logística | 0.7652 ± 0.0125 |
| Árbol de decisión | 0.7738 ± 0.0110 |
| Gradient Boosting | 0.7814 ± 0.0099 |
| SVM con RBF | 0.7827 ± 0.0101 |
| Random Forest | 0.7842 ± 0.0091 |

Los modelos predictivos mejoran claramente al modelo base, que apenas alcanza el desempeño esperado de una predicción basada en la clase mayoritaria. Random Forest presenta la mayor accuracy media y la menor desviación estándar entre los candidatos evaluados, aunque la diferencia frente a SVM y Gradient Boosting es pequeña y comparable con la variación entre folds; por esa razón fue el modelo seleccionado según el criterio definido previamente en el notebook.

Una vez elegido el modelo, `X_test` se utilizó una única vez para la evaluación final: Random Forest obtuvo una accuracy de **0.8010**, frente a **0.5037** del modelo base, con precision, recall y F1-score cercanos a 0.80 en ambas clases.

## Limitaciones y posibles mejoras

### Limitaciones

- **Fuga leve de información dentro de la validación cruzada.** La imputación y el `OneHotEncoder` se ajustaron con todo `X_train` antes de la validación cruzada, y no dentro de cada fold. Esto introduce una fuga leve que afecta por igual a todos los modelos y puede inflar ligeramente las cifras de la tabla de resultados, aunque no debería cambiar el ranking relativo entre ellos. El conjunto `X_test` no se vio afectado, porque la separación se hizo antes de cualquier transformación.
- **Imputación de `Age` no determinista por lote.** La imputación de `Age` muestrea una distribución normal con semilla fija (67), pero la semilla se aplica en cada llamada: la edad asignada a un pasajero con `Age` vacío depende de su posición entre los faltantes de ese lote. Un mismo pasajero puede recibir edades distintas según con quién se prediga. Para esta fase se considera aceptable.
- **El modelo requiere el módulo `preprocesamiento.py`.** `modelo.joblib` incluye una clase propia (`ImputadorTitanic`), por lo que solo puede cargarse si `preprocesamiento.py` está disponible en el path. Además, debe cargarse con la misma versión de `scikit-learn` con que se guardó.

### Mejoras futuras

- Antes de la Fase 2, valorar reemplazar el muestreo aleatorio de `Age` por la mediana, que es determinista y entrega la misma predicción para un mismo pasajero.
- Integrar la imputación y el `OneHotEncoder` dentro de un único `Pipeline` desde el inicio de la comparación, de modo que se ajusten solo con los datos de entrenamiento de cada fold y se elimine la fuga leve.
- Evaluar todos los candidatos con el mismo esfuerzo de búsqueda de hiperparámetros, y hacer una búsqueda más exhaustiva en Random Forest y SVM.
- Evaluar la posibilidad de incorporar métricas adicionales como ROC-AUC.
- Convertir el notebook en scripts de entrenamiento y predicción, a partir de `preprocesamiento.py`.

## Instrucciones para ejecutar el notebook

1. Clonar o descargar este repositorio.
2. Instalar Python 3 y las dependencias utilizadas en el notebook: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn` y `joblib`.
3. Verificar que `Data_Set.csv` se encuentre en la raíz del repositorio y que `preprocesamiento.py` esté en la carpeta `fase-1`, junto al notebook.
4. Abrir `fase-1/notebook.ipynb` en Jupyter Notebook, JupyterLab o Google Colab. En Colab, subir antes `preprocesamiento.py` al panel de archivos para que el notebook pueda importarlo.
5. Ejecutar las celdas en orden, desde la carga y exploración de los datos hasta la evaluación de los candidatos, la selección del modelo y su almacenamiento.

Si se utiliza Jupyter localmente, puede iniciarse con:

```bash
jupyter notebook fase-1/notebook.ipynb
```

## Cómo usar el modelo almacenado (`modelo.joblib`)

`modelo.joblib` es un `Pipeline` de scikit-learn que contiene la imputación de valores faltantes, la transformación logarítmica de los gastos, el one-hot encoding y el Random Forest. Por eso recibe los datos **sin imputar** y devuelve directamente las predicciones.

Para cargarlo:

1. Ubicarse en la carpeta `fase-1`, de modo que `preprocesamiento.py` pueda importarse. Si se ejecuta desde otra carpeta, agregar `fase-1` al path de Python.
2. Usar la misma versión de `scikit-learn` con la que se guardó el modelo.
3. Cargar el modelo y preparar los datos de entrada:

```python
import joblib
import pandas as pd
from preprocesamiento import preparar_datos

modelo = joblib.load("modelo.joblib")

pasajeros = pd.read_csv("../Data_Set.csv")      # datos crudos de Kaggle
X = preparar_datos(pasajeros)                   # deriva Deck desde Cabin y convierte CryoSleep a 0/1

predicciones = modelo.predict(X)                # 1 = transportado, 0 = no transportado
probabilidades = modelo.predict_proba(X)[:, 1]  # probabilidad de ser transportado
```

Requisitos de los datos de entrada:

- `preparar_datos` espera las columnas originales del dataset (`HomePlanet`, `CryoSleep`, `Cabin`, `Age`, `RoomService`, `Spa`, `VRDeck`) y devuelve las 7 variables que usa el modelo: `HomePlanet`, `CryoSleep`, `Deck`, `Age`, `RoomService`, `Spa` y `VRDeck`.
- Los valores faltantes pueden dejarse vacíos: el modelo los imputa con las mismas reglas del notebook, calculadas solo con los datos de entrenamiento.
- Si se prefiere no usar `preparar_datos`, el DataFrame debe contener directamente esas 7 columnas, con `Deck` ya extraído de `Cabin` y `CryoSleep` como 0/1.