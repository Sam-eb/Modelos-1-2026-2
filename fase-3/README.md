# Fase 3: API REST del Titanic Espacial

## Equipo y propósito

**Modelos y Simulación de Sistemas I - Proyecto Spaceship Titanic**

Integrantes: Miguel Angel Avendaño Parra, Miguel Andres Mejía Cuadrado y
Samuel Tabares García.

Esta fase expone mediante HTTP el modelo de la [Fase 2](../fase-2/README.md):
`GET /health` informa el estado del servicio, `POST /predict` predice si un
pasajero fue transportado con la versión configurada y `POST /train` entrena
y guarda una versión nueva sin modificar las existentes.

## Relación con las fases anteriores

`fase-1/` y `fase-2/` se conservan sin cambios. La API no copia lógica de la
Fase 2: la importa tal cual (`src/`, `train.py` y `predict.py`) desde un solo
módulo, [servicio_modelo.py](servicio_modelo.py). Por eso `/predict` aplica la
misma validación de entradas, la misma función `preparar_datos` y el mismo
pipeline guardado que `predict.py`, con las reglas de dominio de `CryoSleep` y
de los gastos y las estadísticas aprendidas solo en entrenamiento. La API
nunca reentrena ni reajusta el modelo para predecir. `/train` usa la misma
función de entrenamiento que `train.py`, con la partición por grupos de viaje
y la validación cruzada de la Fase 2.

| Archivo | Contenido |
|---|---|
| [api.py](api.py) | Fábrica `crear_app` y las tres rutas; `app` para `uvicorn`. |
| [esquemas.py](esquemas.py) | Contrato de entrada y salida (pydantic). |
| [errores.py](errores.py) | Mensajes de validación en español que nombran el campo. |
| [configuracion.py](configuracion.py) | Configuración por variables de entorno. |
| [servicio_modelo.py](servicio_modelo.py) | Único punto que usa el código de la Fase 2. |
| [cliente.py](cliente.py) | Cliente de prueba (solo librería estándar). |
| [solicitudes_curl.md](solicitudes_curl.md) | Ejemplos de `curl` para las tres rutas. |
| [tests/test_api.py](tests/test_api.py) | Pruebas con `pytest` y el `TestClient` de FastAPI. |
| [Dockerfile](Dockerfile) y [Dockerfile.dockerignore](Dockerfile.dockerignore) | Imagen de la API. |

## Datos y modelos

- **Modelos:** las versiones están en `../fase-2/models/`. Se versionan en Git
  `v1` (la que usa la API por defecto) y `v2`, entregadas en la Fase 2.
- **Datos para `/train`:** `../fase-2/data/datos.csv`, ignorado por Git. En una
  copia nueva se crea igual que en la Fase 2, desde la raíz del repositorio:

  ```powershell
  Copy-Item Data_Set.csv fase-2\data\datos.csv
  ```

- **Datos para el cliente:** la primera fila de
  `../fase-2/data/ejemplo_prediccion.csv` (pasajero `0001_01`).

## Ejecución local

Se usa **Python 3.13**, igual que en Docker. Desde la carpeta `fase-3`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

[requirements.txt](requirements.txt) mantiene las versiones de la Fase 2
(los modelos serializados con `joblib` dependen de ellas) y agrega FastAPI,
uvicorn, pydantic y httpx, todas con `==`. El servidor debe iniciarse desde
`fase-3`, porque `api.py` importa los módulos de esa carpeta. La
documentación interactiva queda en `http://127.0.0.1:8000/docs`.

Salida real del servidor al consultar `/health` y ejecutar el cliente:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:59880 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:59882 - "POST /predict HTTP/1.1" 200 OK
```

## Docker

La imagen se construye **desde la raíz del repositorio**, porque necesita el
código de `fase-2` y de `fase-3`:

```powershell
docker build -f fase-3/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) -t modelos-fase3:1.0 .
```

Para iniciar el contenedor, también desde la raíz:

```powershell
docker run -d --rm --name api-fase3 -p 8000:8000 -v "${PWD}/fase-2/data:/app/fase-2/data:ro" -v "${PWD}/fase-2/models:/app/fase-2/models" -e MODELO_VERSION=v1 modelos-fase3:1.0
```

- **Puerto:** el servidor escucha en `0.0.0.0:8000` dentro del contenedor
  (`uvicorn api:app --host 0.0.0.0 --port 8000`), y `-p 8000:8000` lo publica en
  `http://localhost:8000`.
- **Datos:** `fase-2/data` se monta en solo lectura en `/app/fase-2/data`.
- **Modelos:** `fase-2/models` se monta en lectura y escritura en
  `/app/fase-2/models`. Ahí lee `/predict` y escribe `/train`; las versiones
  nuevas quedan en el equipo anfitrión aunque `--rm` elimine el contenedor.
- **Versión:** `-e MODELO_VERSION=...` (ver
  [Seleccionar la versión](#seleccionar-la-versión)).
- **Commit:** `--build-arg GIT_COMMIT=...` lo deja en la imagen, y `/train` lo
  registra en `metadata.json` (`"origen": "variable GIT_COMMIT"`), porque la
  imagen no incluye `.git`.

La imagen usa `python:3.13-slim` y no entrena durante `docker build`. Solo
contiene `fase-2/src`, `train.py`, `predict.py` y `fase-3/`. El
`.dockerignore` excluye `.git`, entornos virtuales, cachés, `docs/`, `fase-1/`,
todos los CSV, los modelos y los archivos con credenciales.

Para detenerlo: `docker stop api-fase3`.

En Git Bash, en lugar de PowerShell, hay que anteponer `MSYS_NO_PATHCONV=1` y
usar `$(pwd -W)` en las rutas de `-v`, como en la Fase 2.

## Operaciones

Los ejemplos de esta sección se ejecutaron el 22-nov-2026 contra el contenedor
iniciado con el comando anterior (imagen construida en el commit `b3f4276`).

### `GET /health`

Indica que la API está en ejecución y si la versión configurada existe.
Responde 200 aunque el modelo falte; el fallo del modelo lo informa
`/predict`.

```powershell
curl.exe -i http://localhost:8000/health
```

```text
HTTP/1.1 200 OK
content-type: application/json

{"estado":"ok","version_configurada":"v1","modelo_disponible":true}
```

### `POST /predict`

Recibe **un registro** JSON con las variables predictoras. No se envía la
variable objetivo `Transported`.

| Campo | Tipo JSON | Unidad o formato | Nulo (`null`) |
|---|---|---|---|
| `HomePlanet` | texto | planeta de origen | moda aprendida |
| `CryoSleep` | booleano | `true`/`false` | regla de gastos y, si no aplica, moda |
| `Cabin` | texto | `"Deck/Num/Side"`; se usa la cubierta | moda de la cubierta |
| `Age` | número ≥ 0 | años | mediana aprendida |
| `RoomService` | número ≥ 0 | créditos de la nave | 0 si `CryoSleep` es `true`; si no, mediana |
| `Spa` | número ≥ 0 | créditos de la nave | igual que `RoomService` |
| `VRDeck` | número ≥ 0 | créditos de la nave | igual que `RoomService` |

- Las siete claves son **obligatorias**, pero admiten `null` como dato
  faltante. La imputación la hace el pipeline guardado con las estadísticas
  aprendidas en el entrenamiento de la versión.
- `Deck` (texto) puede enviarse en lugar de `Cabin`.
- `PassengerId` (texto) es opcional y se devuelve en la respuesta para
  reconocer la observación.
- Las demás claves del CSV original (`Name`, `Destination`, `VIP`, …) se
  ignoran.
- Una **categoría nueva** en `HomePlanet` o en la cubierta (por ejemplo,
  `"Pluto"`) no produce error: el `OneHotEncoder` de la Fase 2
  (`handle_unknown="ignore"`) la codifica con todas sus columnas en 0.

La respuesta trae la predicción (`prediccion`, booleana, y `etiqueta`), la
probabilidad estimada de ser transportado y la **versión** que la produjo.

Solicitud con la fila `0001_01` de `ejemplo_prediccion.csv` (en Git Bash; es
el segundo ejemplo de [solicitudes_curl.md](solicitudes_curl.md)):

```sh
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PassengerId": "0001_01",
    "HomePlanet": "Europa",
    "CryoSleep": false,
    "Cabin": "B/0/P",
    "Age": 39,
    "RoomService": 0,
    "Spa": 0,
    "VRDeck": 0
  }'
```

```text
HTTP/1.1 200 OK
content-type: application/json

{"PassengerId":"0001_01","prediccion":true,"etiqueta":"Transportado","probabilidad":0.5766031273183949,"version":"v1"}
```

Es la misma predicción y probabilidad que da `fase-2/predict.py` con `v1`
para esa fila (`0001_01,1,0.5766031273183949`). Otros casos ejecutados:

| Entrada | Respuesta |
|---|---|
| Fila `0052_01` con `"Age": null` | 200, `false`, probabilidad 0,0448 (igual que `predict.py`) |
| `"HomePlanet": "Pluto"` | 200, `true`, probabilidad 0,5928 |
| Claves adicionales (`Name`, `Transported`) | 200, sin cambios en la predicción |

### Cliente de prueba

[cliente.py](cliente.py) lee la primera fila de
`../fase-2/data/ejemplo_prediccion.csv`, convierte cada columna al tipo del
esquema, la envía a `/predict` e imprime la respuesta. Termina con código 1 si
la API responde un error. Opciones: `--url` (por defecto
`http://localhost:8000/predict`) y `--csv`. Desde `fase-3`:

```powershell
.\.venv\Scripts\python.exe cliente.py
```

```text
{
  "PassengerId": "0001_01",
  "prediccion": true,
  "etiqueta": "Transportado",
  "probabilidad": 0.5766031273183949,
  "version": "v1"
}
```

### Errores

**Campo obligatorio ausente o tipo incompatible → 422.** El mensaje nombra
cada campo con problema. Solicitud sin `Age` (tercer ejemplo de
`solicitudes_curl.md`):

```text
HTTP/1.1 422 Unprocessable Content
content-type: application/json

{"detail":"Entrada inválida. Age: campo obligatorio ausente","errores":[{"campo":"Age","mensaje":"campo obligatorio ausente","tipo":"missing"}]}
```

Con `"Age": "abc"` y `"Spa": -3` en la misma solicitud:

```text
{"detail":"Entrada inválida. Age: debe ser un número; Spa: debe ser mayor o igual que 0","errores":[{"campo":"Age","mensaje":"debe ser un número","tipo":"float_parsing"},{"campo":"Spa","mensaje":"debe ser mayor o igual que 0","tipo":"greater_than_equal"}]}
```

También responden 422 un `CryoSleep` que no sea booleano, la falta de `Cabin`
y de `Deck` a la vez, un cuerpo que no sea un objeto JSON y un JSON mal
formado (indica la posición del error).

**Modelo configurado no disponible → 503.** La API sigue en ejecución y el
mensaje lista las versiones disponibles. Contenedor iniciado con
`-e MODELO_VERSION=v99`:

```text
{"estado":"ok","version_configurada":"v99","modelo_disponible":false}

HTTP 503
{
  "detail": "Modelo no disponible. La versión 'v99' no está disponible en el directorio de modelos configurado (MODELOS_DIR). Versiones disponibles: ['v-docker', 'v1', 'v2', 'v3']. Elija una con la variable de entorno MODELO_VERSION."
}
```

La primera línea es `/health` y el resto, la salida de `cliente.py`. En una
copia nueva del repositorio la lista solo tendría `v1` y `v2`; `v-docker` y
`v3` son versiones locales de verificación que Git ignora.

### Seleccionar la versión

`/predict` usa la versión de la variable de entorno `MODELO_VERSION` (por
defecto `v1`). Las versiones disponibles son las carpetas de `MODELOS_DIR`
que tienen sus tres archivos (`modelo.joblib`, `metrics.json` y
`metadata.json`). Para cambiar de versión se reinicia la API con otro valor,
sin tocar el código:

```powershell
docker stop api-fase3
docker run -d --rm --name api-fase3 -p 8000:8000 -v "${PWD}/fase-2/data:/app/fase-2/data:ro" -v "${PWD}/fase-2/models:/app/fase-2/models" -e MODELO_VERSION=v2 modelos-fase3:1.0
```

Con `v2`, `cliente.py` responde `"probabilidad": 0.5964890852509924` y
`"version": "v2"`. En local se define la variable antes de iniciar `uvicorn`
(`$env:MODELO_VERSION = "v2"` en PowerShell). Si la versión no existe,
`/predict` responde 503 como se mostró antes.

### `POST /train`

Entrena con la configuración de la Fase 2: archivo `DATOS_ENTRENAMIENTO`,
objetivo `TARGET`, semilla `SEMILLA` (42), 20 % de grupos de viaje para
prueba y validación cruzada de 5 folds. Acepta un cuerpo vacío; opcionalmente,
`{"semilla": 123}` cambia la semilla de esa ejecución. Asigna el siguiente
identificador libre (`v3`, `v4`, …), guarda los tres archivos y responde 201
cuando están escritos. **No cambia la versión activa** ni modifica las
versiones existentes.

```powershell
curl.exe -i -X POST http://localhost:8000/train
```

```text
HTTP/1.1 201 Created
content-type: application/json

{"version":"v3","metricas":{"accuracy_test":0.7810550198525241,"accuracy_test_base":0.4815655133295519,"accuracy_cv_mean":0.7897548529096763,"accuracy_cv_std":0.008921019907812832},"version_activa":"v1","mensaje":"Versión 'v3' guardada. /predict sigue usando 'v1'; para usar la nueva, reinicie la API con MODELO_VERSION=v3."}
```

La respuesta tardó unos 8 s. Las métricas coinciden con las de `v1` porque se
usan los mismos datos y la misma semilla. La versión nueva queda en el
almacenamiento persistente, `fase-2/models/v3` en el equipo anfitrión, y
sigue ahí después de `docker stop` (el contenedor se eliminó por `--rm`):

```powershell
Get-ChildItem fase-2\models\v3 | Select-Object Name, Length
```

```text
Name            Length
----            ------
metadata.json     1682
metrics.json       171
modelo.joblib 16018091
```

Su `metadata.json` registra `"hash_commit": "b3f4276"` y
`"origen": "variable GIT_COMMIT"`. `v1` y `v2` conservaron sus archivos y
fechas, y después del entrenamiento `/health` seguía informando
`"version_configurada":"v1"`. Si el CSV de entrenamiento no está, `/train`
responde 503 e indica la variable `DATOS_ENTRENAMIENTO`.

## Configuración

Todas las variables son opcionales; los valores por defecto reproducen la
configuración de la Fase 2 y son rutas relativas al repositorio, iguales en
local y en Docker.

| Variable | Uso | Por defecto |
|---|---|---|
| `MODELOS_DIR` | Carpeta de las versiones | `fase-2/models` |
| `MODELO_VERSION` | Versión que usa `/predict` | `v1` |
| `DATOS_ENTRENAMIENTO` | CSV que usa `/train` | `fase-2/data/datos.csv` |
| `TARGET` | Variable objetivo | `Transported` |
| `SEMILLA` | Semilla de `/train` | `42` |
| `TEST_SIZE` | Proporción de grupos de prueba | `0.2` |
| `CV_FOLDS` | Folds de la validación cruzada (0 la omite) | `5` |

Un valor numérico inválido detiene el arranque con un mensaje que nombra la
variable.

## Pruebas

Desde la carpeta `fase-3`, con el entorno de la sección
[Ejecución local](#ejecución-local):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

[pytest.ini](pytest.ini) agrega `fase-3` y `../fase-2` al `pythonpath`. Las
pruebas no usan red ni las versiones del repositorio: cada una entrena en
`tmp_path` un Random Forest pequeño (20 árboles) con 40 pasajeros sintéticos,
lo guarda como `v1` con el almacenamiento de la Fase 2 y crea la API con el
`TestClient` de FastAPI.

| Prueba | Qué comprueba |
|---|---|
| `test_health_responde_ok` | `/health` responde 200 con `estado == "ok"`. |
| `test_predict_entrada_valida` | Un registro válido responde 200 con `prediccion` y `version`, y la versión es la configurada. |
| `test_predict_campo_obligatorio_ausente` | El mismo registro sin `Age` responde 422 y el mensaje nombra `Age`. |
| `test_predict_coincide_con_predict_py` | La predicción y la probabilidad de la API coinciden con las de `predict.py`. |
| `test_predict_nulo_permitido` | `"Age": null` responde 200. |
| `test_predict_tipo_incompatible` | `"Age": "abc"` responde 422 y nombra `Age`. |
| `test_predict_modelo_inexistente` | Con `v99` configurada, `/predict` responde 503 y menciona `MODELO_VERSION`. |
| `test_train_crea_version_sin_tocar_anteriores` | `/train` responde 201 con `v2`, la activa sigue siendo `v1` y sus archivos no cambian. |

Las tres primeras son las que pide la especificación; las demás cubren los
otros requisitos. Salida real:

```text
tests/test_api.py::test_health_responde_ok PASSED                        [ 12%]
tests/test_api.py::test_predict_entrada_valida PASSED                    [ 25%]
tests/test_api.py::test_predict_campo_obligatorio_ausente PASSED         [ 37%]
tests/test_api.py::test_predict_coincide_con_predict_py PASSED           [ 50%]
tests/test_api.py::test_predict_nulo_permitido PASSED                    [ 62%]
tests/test_api.py::test_predict_tipo_incompatible PASSED                 [ 75%]
tests/test_api.py::test_predict_modelo_inexistente PASSED                [ 87%]
tests/test_api.py::test_train_crea_version_sin_tocar_anteriores PASSED   [100%]
======================== 8 passed, 1 warning in 3.77s =========================
```

El aviso es una deprecación de Starlette sobre el uso de `httpx` en el
`TestClient`; no afecta a las pruebas. La comprobación manual de `/train` es
la de la sección [`POST /train`](#post-train). Las pruebas de la Fase 2
siguen pasando sin cambios (`python -m pytest` desde `fase-2`: 13 pruebas).

## Limitaciones

- `/predict` recibe un registro por solicitud; no acepta listas.
- `/train` es síncrono: la solicitud espera al entrenamiento completo (unos
  8 s en Docker con los datos reales). Si dos entrenamientos piden el mismo
  identificador a la vez, el segundo reintenta con el siguiente.
- La probabilidad es la estimada por el Random Forest; no está calibrada.
- El guardado de versiones de la Fase 2 deja una carpeta temporal vacía y
  oculta (por ejemplo, `fase-2/models/.v3.xxxxxxxx`) junto a cada versión
  nueva. La API la ignora al listar versiones y Git no la versiona; no se
  corrigió porque `fase-2` está congelada.
- En Git Bash, `cliente.py` puede mostrar mal las tildes por la codificación
  de la consola; en PowerShell se ven bien.
