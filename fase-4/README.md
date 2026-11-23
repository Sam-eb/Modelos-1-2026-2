# Fase 4: monitoreo del servicio

## Equipo y propósito

**Modelos y Simulación de Sistemas I - Proyecto Spaceship Titanic**

Integrantes: Miguel Angel Avendaño Parra, Miguel Andres Mejía Cuadrado y
Samuel Tabares García.

Esta fase agrega **registro y monitoreo** a la API de la
[Fase 3](../fase-3/README.md).  Cada solicitud a `/predict` (exitosa o
fallida) queda registrada en un CSV persistente y un script independiente
calcula los indicadores del servicio.

## Relación con las fases anteriores

`fase-1/`, `fase-2/` y `fase-3/` se conservan sin cambios.  La Fase 4 importa
la fábrica `crear_app` de la Fase 3 y le conecta un **observador** que
registra cada intento de predicción; no modifica ni copia lógica de las fases
anteriores.

| Archivo | Contenido |
|---|---|
| [api_monitoreo.py](api_monitoreo.py) | Importa `crear_app` de la Fase 3 y conecta el registro. |
| [registro.py](registro.py) | Escribe una fila por solicitud en el CSV (modo agregar). |
| [monitoreo.py](monitoreo.py) | Lee el registro y muestra total y % de fallidas. |
| [tests/test_monitoreo.py](tests/test_monitoreo.py) | Pruebas con `pytest` y `TestClient`. |
| [muestra_registro.csv](muestra_registro.csv) | Muestra del registro del ensayo real. |
| [Dockerfile](Dockerfile) y [Dockerfile.dockerignore](Dockerfile.dockerignore) | Imagen de la API con registro. |

## Ejecución local

Desde la carpeta `fase-4`, con Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api_monitoreo:app --host 127.0.0.1 --port 8000
```

El registro se escribe en `fase-4/logs/predicciones.csv` por defecto.

## Docker

La imagen se construye **desde la raíz del repositorio**:

```powershell
docker build -f fase-4/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) -t modelos-fase4:1.0 .
```

Para iniciar el contenedor:

```powershell
docker run -d --rm --name api-fase4 -p 8000:8000 -v "${PWD}/fase-2/data:/app/fase-2/data:ro" -v "${PWD}/fase-2/models:/app/fase-2/models" -v "${PWD}/fase-4/logs:/app/fase-4/logs" -e MODELO_VERSION=v1 modelos-fase4:1.0
```

- **Registro persistente:** `-v .../fase-4/logs:/app/fase-4/logs` monta la
  carpeta de registro.  El CSV persiste tras `docker run --rm`.
- **Datos y modelos:** mismos montajes que la Fase 3.

## Registro de solicitudes

Cada solicitud a `/predict` genera **exactamente una fila** en el CSV con:

| Campo | Descripción |
|---|---|
| `fecha_hora` | Marca ISO 8601 en UTC. |
| `estado_http` | Código HTTP de la respuesta (200, 422, 503). |
| `version` | Versión del modelo usada (vacío si no se usó modelo). |
| `prediccion` | `true` o `false` (vacío si hubo error). |

El encabezado se escribe solo al crear el archivo.  El modo es agregar: las
filas se conservan entre reinicios de la API.

## Script de monitoreo

```powershell
.\.venv\Scripts\python.exe monitoreo.py
```

Muestra el total de solicitudes y el porcentaje de fallidas (HTTP 4xx/5xx).
Con registro vacío o inexistente: total 0 y «sin solicitudes».

> [!IMPORTANT]
> El porcentaje de solicitudes fallidas mide la **disponibilidad del
> servicio** (errores de validación y de infraestructura), **no** la calidad
> predictiva del modelo.  Una solicitud con predicción incorrecta pero
> respuesta 200 no cuenta como fallo del servicio.

## Reporte

### Ensayo real

Sobre un registro vacío se enviaron 3 solicitudes a `/predict`:

1. **Válida** – pasajero `0001_01` con todos los campos → 200.
2. **Válida** – mismo pasajero → 200.
3. **Inválida** – sin el campo `Age` → 422.

Contenido del registro (`muestra_registro.csv`):

```csv
fecha_hora,estado_http,version,prediccion
2026-11-22T21:10:00+00:00,200,v1,true
2026-11-22T21:10:05+00:00,200,v1,true
2026-11-22T21:10:10+00:00,422,,
```

Salida de `monitoreo.py`:

```text
Total de solicitudes: 3
Solicitudes fallidas: 33.3%
```

**Interpretación:** de las 3 solicitudes, 1 fue rechazada por falta de un
campo obligatorio (`Age`), lo que representa el 33,3 % de fallos del
servicio.  Las 2 solicitudes exitosas recibieron predicción con la versión
`v1`.  El porcentaje de fallos refleja errores en la solicitud del cliente, no
problemas del modelo ni del servidor.

### Tabla de versiones

| Versión | Algoritmo | accuracy (test) | accuracy (CV) | Semilla | Python | scikit-learn | Commit |
|---|---|---|---|---|---|---|---|
| `v1` | RandomForestClassifier | 0.7811 | 0.7898 ± 0.0089 | 42 | 3.13 | 1.9.1 | `1ac70b4` |
| `v2` | RandomForestClassifier | 0.7812 | 0.7867 ± 0.0068 | 123 | 3.13 | 1.9.1 | `1ac70b4` |

Ambas versiones usan las mismas 7 variables (`HomePlanet`, `CryoSleep`,
`Deck`, `Age`, `RoomService`, `Spa`, `VRDeck`), el mismo pipeline de
preprocesamiento y la misma partición por grupos de viaje.

### Versión activa y cómo cambiarla

La API usa **`v1`** por defecto.  Para usar `v2` (o cualquier otra versión
disponible), basta con cambiar la variable de entorno:

```powershell
# Local
$env:MODELO_VERSION="v2"
uvicorn api_monitoreo:app --host 127.0.0.1 --port 8000

# Docker
docker run -d --rm --name api-fase4 -p 8000:8000 -v "${PWD}/fase-2/data:/app/fase-2/data:ro" -v "${PWD}/fase-2/models:/app/fase-2/models" -v "${PWD}/fase-4/logs:/app/fase-4/logs" -e MODELO_VERSION=v2 modelos-fase4:1.0
```

No se modifica código ni se reentrena: el cambio de versión es inmediato.

### Política de reentrenamiento

El reentrenamiento se considera cuando se dispone de un lote significativo de
pasajeros nuevos con `Transported` conocido (por ejemplo, 500 o más registros
adicionales).  Cualquier integrante del equipo puede ejecutar `POST /train`
para generar una versión candidata; la decisión de activarla requiere que su
accuracy sobre los mismos casos de prueba reservados (misma partición por
grupos, misma proporción) supere a la versión activa y al modelo base.  Si la
nueva versión cambia la semilla o los datos de prueba respecto a la anterior,
esa diferencia se debe documentar antes de interpretar un cambio en las
métricas como mejora real.

## Pruebas

Desde la carpeta `fase-4`, con el entorno de la sección
[Ejecución local](#ejecución-local):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

| Prueba | Qué comprueba |
|---|---|
| `test_registro_dos_validas_una_invalida` | 2 válidas + 1 inválida → 3 filas y 33,3 %. |
| `test_registro_vacio_sin_solicitudes` | Registro vacío → total 0 y «sin solicitudes». |
| `test_rechazo_genera_una_sola_fila` | Un rechazo 422 genera exactamente una fila. |
| `test_registro_503_modelo_inexistente` | Modelo inexistente → 503 con una fila. |
| `test_registro_persiste_entre_ejecuciones` | Dos ejecuciones → filas acumuladas. |

## Configuración

Todas las variables de la Fase 3 siguen vigentes, más una adicional:

| Variable | Uso | Por defecto |
|---|---|---|
| `REGISTRO_PREDICCIONES` | Ruta del CSV de registro | `logs/predicciones.csv` |

## Limitaciones

- El registro es un CSV en disco; no soporta escritura concurrente desde
  múltiples procesos (sí es seguro entre hilos del mismo proceso con
  `threading.Lock`).
- `monitoreo.py` lee todo el archivo en memoria; con millones de filas sería
  necesario un procesamiento incremental.
- El porcentaje de fallos no distingue entre errores del cliente (422) y
  errores del servidor (503); ambos cuentan como solicitudes fallidas.
