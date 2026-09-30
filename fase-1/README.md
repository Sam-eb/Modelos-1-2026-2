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

Los modelos predictivos mejoran claramente al modelo base, que apenas alcanza el desempeño esperado de una predicción basada en la clase mayoritaria. Random Forest presenta la mayor accuracy media entre los candidatos evaluados, aunque la diferencia frente a SVM y Gradient Boosting es pequeña y comparable con la variación entre folds. Por esto todavía no se considera seleccionado de forma definitiva: el conjunto `X_test` debe utilizarse una única vez, después de decidir cuál modelo se conservará.

## Instrucciones para ejecutar el notebook

1. Clonar o descargar este repositorio.
2. Instalar Python 3 y las dependencias utilizadas en el notebook: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib` y `seaborn`.
3. Verificar que `Data_Set.csv` se encuentre en la raíz del repositorio.
4. Abrir `fase-1/notebook.ipynb` en Jupyter Notebook, JupyterLab o Google Colab.
5. Ejecutar las celdas en orden, desde la carga y exploración de los datos hasta la evaluación de los candidatos.

Si se utiliza Jupyter localmente, puede iniciarse con:

```bash
jupyter notebook fase-1/notebook.ipynb
```