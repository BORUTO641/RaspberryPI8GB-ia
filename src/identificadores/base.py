"""Contrato común. Cualquier identificador (el actual o la IA futura) debe cumplirlo."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Deteccion:
    etiqueta: str                      # p. ej. "rostro", "persona", "Juan Pérez"
    confianza: float                   # 0.0 a 1.0
    caja: tuple[int, int, int, int]    # x, y, ancho, alto en píxeles
    extra: dict = field(default_factory=dict)  # datos del modelo; se guardan en el JSON
    privado: dict = field(default_factory=dict, repr=False)  # solo para dibujar; NO se guarda

    def a_json(self) -> dict:
        return {"etiqueta": self.etiqueta, "confianza": self.confianza,
                "caja": list(self.caja), "extra": self.extra}


class Identificador(ABC):
    nombre: str = "base"

    @abstractmethod
    def identificar(self, frame) -> list[Deteccion]:
        """Recibe un frame BGR (numpy) y devuelve lo que encontró."""

    def recalibrar(self):
        """Olvida lo aprendido de la persona (p. ej. al cambiar de persona). Opcional."""

    def cerrar(self):
        """Libera recursos (modelos, sesiones). Opcional."""
