# El extraño caso del Titanic Espacial

Modelos y Simulación de Sistemas I | Proyecto Spaceship Titanic
Integrantes: Miguel Angel Avendaño Parra, Miguel Andres Mejía Cuadrado y Samuel Tabares García.

Para la documentación completa, metodología, resultados y guía de uso del modelo, consultar [fase-1/README.md](fase-1/README.md) y [fase-2/README.md](fase-2/README.md).

## Estructura del repositorio

```
Data_Set.csv              <- Datos originales de Kaggle Spaceship Titanic
README.md                 <- Resumen del repositorio
fase-1/
├── README.md             <- Documentación detallada y reporte de resultados
├── notebook.ipynb        <- Notebook principal con flujo metodológico completo
├── notebook_outputs.ipynb<- Notebook con salidas y gráficas ejecutadas
├── preprocesamiento.py   <- Módulo con clases y transformadores de datos (Pipeline)
└── modelo.joblib         <- Pipeline entrenado, optimizado y serializado
fase-2/
├── README.md              <- Guía de scripts, pruebas, versiones y Docker
├── train.py               <- Entrenamiento y evaluación versionada
├── predict.py             <- Predicción con una versión guardada
├── requirements.txt       <- Dependencias exactas de Python 3.13
├── Dockerfile             <- Imagen reproducible de la aplicación
├── tests/                 <- Pruebas automatizadas con pytest
├── src/                   <- Lógica compartida reutilizable
├── data/                  <- CSV de ejemplo de predicción
└── models/                <- Versiones v1 y v2 con métricas y metadatos
```

`Data_Set.csv` es compartido por ambas fases. Las dependencias de la Fase 1
están en [fase-1/requirements.txt](fase-1/requirements.txt); las de la Fase 2
están en [fase-2/requirements.txt](fase-2/requirements.txt).
