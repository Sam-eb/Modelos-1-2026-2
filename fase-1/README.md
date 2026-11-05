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

## Fuente del conjunto de datos y entorno reproducible

El conjunto de datos corresponde a la competencia **Spaceship Titanic** de Kaggle:

https://www.kaggle.com/competitions/spaceship-titanic

El archivo utilizado es `Data_Set.csv`, ubicado en la raíz del repositorio (8.693 registros y 14 variables originales). 

> [!NOTE]
> **Carga reproducible y versiones fijas:** Para asegurar total reproducibilidad y evitar depender de ramas remotas mutables (como `develop` o `main`), el código carga prioritariamente el archivo local (`Data_Set.csv` / `../Data_Set.csv`). En entornos externos (como Google Colab sin clonación directa), se utiliza un enlace directo a un **commit inmutable** (SHA: `fa35aef4627d3fa30536efce7db8f352136e6b4f`).

### Versiones de software compatibles
El proyecto ha sido desarrollado y verificado en el siguiente entorno:
- **Python**: `>= 3.10` (verificado en `3.13.3`)
- **scikit-learn**: `>= 1.5.0` (verificado en `1.9.1`)
- **pandas**: `>= 2.0.0` (verificado en `2.3.0`)
- **numpy**: `>= 1.26.0` (verificado en `2.3.0`)
- **scipy**: `>= 1.12.0` (verificado en `1.16.0`)
- **joblib**: `>= 1.3.0` (verificado en `1.6.0`)
- **matplotlib** y **seaborn** para visualizaciones.

## Estructura de la entrega

```
Data_Set.csv
README.md
fase-1/
├── README.md
├── notebook.ipynb
├── notebook_outputs.ipynb
├── preprocesamiento.py
└── modelo.joblib
```

## Análisis del Dataset y Valores Faltantes

El conjunto de datos contiene 8.693 registros. Al realizar el análisis de valores faltantes por columna:
- Las columnas con nulos presentan porcentajes entre **2,059 % y 2,496 %** (por ejemplo: `Age` con 2,059 %, `FoodCourt` con 2,105 %, `Spa` con 2,105 %, `Cabin` con 2,289 % y `CryoSleep` con 2,496 %).
- **Aclaración sobre el requisito de faltantes:** Ninguna columna con valores faltantes se ubica dentro del rango estricto de 0,1 % a 2,0 %; todas superan ligeramente el 2 %. Se aclara explícitamente que estas proporciones no deben redondearse al 2 % para afirmar cumplimiento estricto.
- Dado que el porcentaje de pérdida sigue siendo bajo (< 2,5 %) y las variables aportan información predictiva fundamental, se descartó la eliminación de filas completas (que descartaría más del 20 % de los pasajeros) y se optó por un esquema de imputación fundamentado en reglas de dominio y estadística robusta.

## Objetivo del modelo y preparación de datos

Clasificar, según los datos registrados del pasajero, si fue transportado a otra dimensión mediante la variable objetivo booleana `Transported`.

La preparación de datos incluye:
1. **Separación por grupos de viaje (`StratifiedGroupKFold`)**: Con 5 pliegues y semilla fija (67), garantizando que todos los miembros de un mismo grupo (`Group`, extraído de `PassengerId`) queden exclusivamente en entrenamiento o en prueba, evitando fuga entre pasajeros relacionados.
2. **Imputación de valores faltantes determinista**:
   - `HomePlanet` y `Deck`: imputados con la moda aprendida en entrenamiento.
   - Reglas de dominio para gastos y sueño criogénico: pasajeros con `CryoSleep == True` tienen gastos imputados en 0 (dormidos no consumen); pasajeros con `CryoSleep` nulo pero con algún gasto > 0 se imputan con `CryoSleep = 0` (quien gasta está despierto); el resto de `CryoSleep` se imputa con la moda.
   - `Age`: **Imputación por mediana aprendida en entrenamiento**. Se sustituyó el muestreo estocástico normal previo por la mediana fija de entrenamiento. Esto resuelve la inestabilidad donde predecir un lote frente a un registro individual cambiaba clasificaciones y generaba discrepancias de probabilidad de hasta 0,379. La mediana asegura determinismo estricto e idéntico comportamiento en una futura API.
   - Gastos restantes (`RoomService`, `Spa`, `VRDeck`): imputados con la mediana de entrenamiento.
3. **Transformaciones y codificación**:
   - Transformación `log1p` sobre las variables de gasto para suavizar su asimetría.
   - Codificación One-Hot (`OneHotEncoder(handle_unknown="ignore")`) sobre las variables categóricas `HomePlanet` y `Deck`.
   - Escalamiento numérico (`StandardScaler`) integrado dentro de pipelines para modelos sensibles a distancias geométricas (Regresión Logística y SVM).

## Prevención de fuga de información (Data Leakage)

En atención a la retroalimentación de la Fase 1, se corrigió la validación para eliminar cualquier intervención o fuga de información:
- **Partición antes de la selección de variables**: Las asociaciones supervisadas con `Transported` (correlaciones de Pearson, $p$-valores, V de Cramér y correlaciones biseriales) se restringen estrictamente a los datos de entrenamiento (`X_train`, `y_train`). El conjunto de prueba `X_test` permanece totalmente aislado y ciego a cualquier decisión de filtrado o ingeniería de características.
- **Preprocesamiento anidado en cada fold de Cross-Validation**: Se eliminó el ajuste previo de imputación y One-Hot sobre todo `X_train`. Cada candidato se evalúa como un `Pipeline` completo que encapsula `crear_preprocesamiento()` (o los transformadores correspondientes). Dentro de cada pliegue de validación cruzada (`StratifiedGroupKFold`), el imputador y el encoder se ajustan (*fit*) exclusivamente sobre los pliegues de entrenamiento y se aplican (*transform*) sobre el pliegue de validación.
- **Corrección sobre el ranking de modelos**: Se retira la afirmación previa de que *"la fuga no altera el orden de los modelos"*; la literatura y la práctica metodológica demuestran que el preprocesamiento previo a la validación cruzada contamina las estimaciones y puede distorsionar la comparación relativa. La evaluación mediante pipelines anidados fold a fold proporciona ahora una comparación metodológicamente válida e imparcial.

## Algoritmos evaluados

Se evaluaron seis candidatos dentro de la misma estrategia de validación cruzada agrupada y estratificada (5 folds, semilla 67):
- **Modelo base**: `DummyClassifier(strategy="most_frequent")`.
- **Regresión logística**: Pipeline con preprocesamiento, `StandardScaler` y regularización L2 (`max_iter=1000`).
- **Árbol de decisión**: Pipeline con preprocesamiento y `max_depth=7`.
- **Gradient Boosting**: Pipeline con preprocesamiento, `n_estimators=100`, `learning_rate=0.05` y `max_depth=3`.
- **SVM con kernel RBF**: Pipeline con preprocesamiento, `StandardScaler` y optimización de $C$ y $\gamma$.
- **Random Forest**: Pipeline con preprocesamiento y búsqueda de hiperparámetros (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).

## Principales resultados obtenidos

### Validación cruzada anidada sobre `X_train` (sin fuga de información)

| Modelo | Accuracy CV (Media ± Desv. Est.) | F1 False | F1 True |
| :--- | :---: | :---: | :---: |
| Modelo base | 0.5036 ± 0.0002 | 0.0000 | 0.6700 |
| Regresión logística | 0.7652 ± 0.0125 | 0.7651 | 0.7650 |
| Árbol de decisión | 0.7725 ± 0.0104 | 0.7500 | 0.7900 |
| Gradient Boosting | 0.7813 ± 0.0107 | 0.7700 | 0.8000 |
| SVM con RBF | 0.7827 ± 0.0099 | 0.7700 | 0.7900 |
| **Random Forest** | **0.7856 ± 0.0089** | **0.7800** | **0.7900** |

Random Forest obtiene la mayor accuracy media en validación cruzada y la menor dispersión entre folds, además de mostrar un equilibrio sólido entre ambas clases.

### Distinción de métricas en conjunto de prueba (`X_test`)

Para transparentar los resultados y atender la observación de la evaluación:

1. **Modelo evaluado preliminarmente en notebook (Entrega 1)**: Accuracy de **0.8010**. Esta cifra se obtuvo al evaluar Random Forest sobre datos transformados previamente en bloque con imputación estocástica de edad.
2. **Pipeline integrado guardado en Entrega 1 (`modelo.joblib`)**: Accuracy de **0.8005**. Obtenida al entrenar `pipeline_completo` con `RandomizedSearchCV` directamente sobre los datos crudos.
3. **Pipeline final corregido (Entrega 2)**: Accuracy de **0.7993** (~0.80) en `X_test` y **0.7850** en CV interno. Este pipeline incorpora la imputación determinista por mediana en `Age`, asegurando estabilidad absoluta entre inferencias individuales y por lotes, y preprocesamiento fold a fold.

> [!IMPORTANT]
> **Uso del conjunto de prueba:** Se documenta que en la versión exploratoria previa se consultó `X_test` en dos etapas distintas. Metodológicamente, el conjunto de prueba se mantiene estrictamente bloqueado y debe evaluarse una única vez al finalizar el pipeline definitivo seleccionado.

## Conclusiones del análisis exploratorio (EDA) moderadas

En concordancia con las buenas prácticas estadísticas:
- **Interpretación del $p$-valor**: Un $p$-valor no significativo ($p \ge 0,05$, como en `ShoppingMall` frente a `Transported`) **no demuestra ausencia de relación**; únicamente indica falta de evidencia estadística suficiente bajo el modelo evaluado para rechazar la hipótesis nula.
- **Relación entre edad cero y consumos**: Que el 100 % de los pasajeros registrados con edad 0 reporte gasto cero en todos los servicios a bordo es **evidencia indicativa y coherente** con infantes que no consumen servicios, pero no constituye por sí solo una demostración concluyente de que la edad sea biológicamente exacta.

## Git Flow y Colaboración del Equipo

- El repositorio opera bajo el modelo **Git Flow**, con ramas principales `main` (producción) y `develop` (integración), complementadas con ramas temáticas (`feature/*`, `docs/*`, `fix/*`).
- Se han integrado más de 17 Pull Requests con participación activa y verificable de los tres integrantes (Miguel Avendaño, Miguel Mejía y Samuel Tabares).
- **Regla formal de revisión de código:** Para las siguientes entregas, ningún Pull Request se fusiona a `develop` o `main` sin la revisión formal y aprobación explícita registrada en GitHub por parte de al menos otro integrante del equipo.

## Instrucciones para ejecutar el notebook

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Sam-eb/Modelos-1-2026-2.git
   cd Modelos-1-2026-2
   ```
2. Instalar las dependencias recomendadas:
   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
   ```
3. Verificar la presencia de `Data_Set.csv` en la raíz y de `fase-1/preprocesamiento.py` en la carpeta `fase-1`.
4. Iniciar Jupyter y ejecutar `fase-1/notebook.ipynb`:
   ```bash
   jupyter notebook fase-1/notebook.ipynb
   ```

## Cómo usar el modelo almacenado (`modelo.joblib`)

`modelo.joblib` es un objeto `Pipeline` de scikit-learn serializado que integra la imputación por reglas de dominio y mediana, la transformación `log1p`, el `OneHotEncoder` y el clasificador `RandomForestClassifier`. Recibe los datos con valores faltantes y devuelve directamente predicciones deterministas.

```python
import joblib
import pandas as pd
import sys
sys.path.append("fase-1")  # Permitir importación de preprocesamiento.py
from preprocesamiento import preparar_datos

# 1. Cargar el pipeline serializado
modelo = joblib.load("fase-1/modelo.joblib")

# 2. Cargar datos crudos
pasajeros = pd.read_csv("Data_Set.csv")
X = preparar_datos(pasajeros)  # Extrae Deck y convierte CryoSleep

# 3. Inferencia individual o por lote (comportamiento idéntico y determinista)
predicciones = modelo.predict(X)                # 1 = Transportado, 0 = No transportado
probabilidades = modelo.predict_proba(X)[:, 1]  # Probabilidad de transporte

print("Predicciones generadas:", len(predicciones))
```
