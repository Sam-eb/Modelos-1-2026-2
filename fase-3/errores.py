"""Mensajes de error de validación en español, con el campo que falla."""

from typing import Any

from fastapi.exceptions import RequestValidationError

_MENSAJES_ERROR = {
    "missing": "campo obligatorio ausente",
    "float_parsing": "debe ser un número",
    "float_type": "debe ser un número",
    "bool_parsing": "debe ser booleano (true/false)",
    "bool_type": "debe ser booleano (true/false)",
    "string_type": "debe ser texto",
    "greater_than_equal": "debe ser mayor o igual que 0",
    "int_parsing": "debe ser un entero",
    "int_type": "debe ser un entero",
    "extra_forbidden": "campo no permitido",
    "model_attributes_type": "el cuerpo debe ser un objeto JSON",
    "dict_type": "el cuerpo debe ser un objeto JSON",
}


def describir_errores(error: RequestValidationError) -> list[dict[str, Any]]:
    """Convierte los errores de pydantic en mensajes en español con el campo."""
    errores = []
    for detalle in error.errors():
        ubicacion = [str(parte) for parte in detalle.get("loc", ()) if parte != "body"]
        campo = ".".join(ubicacion) or None
        tipo = detalle.get("type", "")
        if tipo == "json_invalid":
            # En este caso loc trae la posición del error dentro del texto.
            mensaje = f"el cuerpo no es un JSON válido (posición {campo})"
            campo = None
        elif tipo == "value_error":
            mensaje = str(detalle.get("ctx", {}).get("error", detalle.get("msg", "")))
        else:
            mensaje = _MENSAJES_ERROR.get(tipo, detalle.get("msg", "valor inválido"))
            if tipo == "missing" and campo == "Cabin":
                mensaje += " (envíe Cabin o, en su lugar, Deck)"
        errores.append({"campo": campo, "mensaje": mensaje, "tipo": tipo})
    return errores


def cuerpo_error_validacion(error: RequestValidationError) -> dict[str, Any]:
    """Cuerpo de la respuesta 422: un resumen legible y el detalle por campo."""
    errores = describir_errores(error)
    resumen = "; ".join(
        f"{e['campo']}: {e['mensaje']}" if e["campo"] else e["mensaje"] for e in errores
    )
    return {"detail": f"Entrada inválida. {resumen}", "errores": errores}
