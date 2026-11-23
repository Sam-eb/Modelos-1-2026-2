# Ejemplos de solicitudes

Con la API iniciada en `http://localhost:8000`:

```sh
# Se espera HTTP 200 y el estado del servicio y la disponibilidad de la versión.
curl -i http://localhost:8000/health
```

```sh
# Se espera HTTP 200 con predicción, etiqueta, probabilidad, versión y PassengerId.
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

```sh
# Se espera HTTP 422; la respuesta identifica Age como campo obligatorio ausente.
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PassengerId": "0001_01",
    "HomePlanet": "Europa",
    "CryoSleep": false,
    "Cabin": "B/0/P",
    "RoomService": 0,
    "Spa": 0,
    "VRDeck": 0
  }'
```

```sh
# Se espera HTTP 201 con el identificador de la nueva versión y sus métricas.
curl -i -X POST http://localhost:8000/train
```
