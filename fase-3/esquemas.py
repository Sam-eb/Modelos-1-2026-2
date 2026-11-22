"""Contrato de la API: cuerpos de las solicitudes y de las respuestas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

ETIQUETAS = {True: "Transportado", False: "No transportado"}


class Pasajero(BaseModel):
    """Registro de un pasajero con las variables que usa el modelo.

    Todas las claves son obligatorias, pero admiten `null` como dato faltante:
    el pipeline guardado lo imputa con las reglas aprendidas en entrenamiento.
    La cubierta se toma de `Cabin` ("Deck/Num/Side") o de `Deck`; basta una de
    las dos. Otras columnas del CSV original (p. ej. `Name`) se ignoran.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "PassengerId": "0001_01",
                "HomePlanet": "Europa",
                "CryoSleep": False,
                "Cabin": "B/0/P",
                "Age": 39.0,
                "RoomService": 0.0,
                "Spa": 0.0,
                "VRDeck": 0.0,
            }
        },
    )

    PassengerId: str | None = Field(
        default=None, description="Identificador opcional; se devuelve en la respuesta."
    )
    HomePlanet: str | None = Field(..., description="Planeta de origen (texto).")
    CryoSleep: bool | None = Field(..., description="Viaja en sueño criogénico.")
    Cabin: str | None = Field(
        ..., description='Cabina "Deck/Num/Side"; se usa su cubierta. Opcional si se envía Deck.'
    )
    Deck: str | None = Field(default=None, description="Cubierta, si no se envía Cabin.")
    Age: float | None = Field(..., ge=0, description="Edad en años (>= 0).")
    RoomService: float | None = Field(
        ..., ge=0, description="Gasto en servicio a la habitación (créditos, >= 0)."
    )
    Spa: float | None = Field(..., ge=0, description="Gasto en el spa (créditos, >= 0).")
    VRDeck: float | None = Field(
        ..., ge=0, description="Gasto en la cubierta de realidad virtual (créditos, >= 0)."
    )

    @model_validator(mode="before")
    @classmethod
    def _deck_reemplaza_cabin(cls, datos: Any) -> Any:
        # Si llega Deck sin Cabin, Cabin no es necesaria: la Fase 2 usa Deck
        # cuando la columna existe. Si no llega ninguna, Cabin queda ausente y
        # se informa junto con los demás campos obligatorios.
        if isinstance(datos, dict) and "Cabin" not in datos and "Deck" in datos:
            datos = {**datos, "Cabin": None}
        return datos

    def a_registro(self) -> dict[str, Any]:
        """Diccionario para el modelo; Deck solo se incluye si se envió."""
        registro = self.model_dump()
        if "Deck" not in self.model_fields_set:
            registro.pop("Deck")
        return registro


class RespuestaPrediccion(BaseModel):
    PassengerId: str | None = None
    prediccion: bool = Field(description="True si el pasajero fue transportado.")
    etiqueta: str
    probabilidad: float | None = Field(
        default=None, description="Probabilidad estimada de ser transportado."
    )
    version: str = Field(description="Versión del modelo que generó la predicción.")


class SolicitudEntrenamiento(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semilla: int | None = Field(
        default=None, description="Semilla; si se omite, se usa la configurada (SEMILLA)."
    )


class RespuestaEntrenamiento(BaseModel):
    version: str
    metricas: dict[str, float | None]
    version_activa: str
    mensaje: str
