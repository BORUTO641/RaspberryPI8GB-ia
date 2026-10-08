from .base import Deteccion, Identificador


MODOS = ["gestos", "rostros", "ia"]


def crear_identificador(modo: str) -> Identificador:
    if modo == "gestos":
        from .gestos_mediapipe import IdentificadorGestos
        return IdentificadorGestos()
    if modo == "rostros":
        from .rostros_opencv import IdentificadorRostros
        return IdentificadorRostros()
    if modo == "ia":
        from .ia_local import IdentificadorIALocal
        return IdentificadorIALocal()
    raise ValueError(f"Modo de identificador desconocido: {modo!r}")


__all__ = ["Deteccion", "Identificador", "MODOS", "crear_identificador"]
