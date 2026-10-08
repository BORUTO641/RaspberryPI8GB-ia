"""Identificador provisional: detección de rostros con Haar Cascade (incluido en OpenCV).

No reconoce QUIÉN es la persona, solo que hay un rostro. Sirve para probar todo el
flujo (cámara -> detección -> archivo de salida) mientras llega la IA.
"""
import cv2

from .base import Deteccion, Identificador


class IdentificadorRostros(Identificador):
    nombre = "rostros_opencv"

    def __init__(self):
        ruta = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.clasificador = cv2.CascadeClassifier(ruta)
        if self.clasificador.empty():
            raise RuntimeError(f"No se pudo cargar el clasificador: {ruta}")

    def identificar(self, frame) -> list[Deteccion]:
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris)
        cajas = self.clasificador.detectMultiScale(
            gris, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )
        # Haar no da una probabilidad real; se pone 1.0 y se indica en `extra`.
        return [
            Deteccion("rostro", 1.0, tuple(int(v) for v in c), {"confianza_real": False})
            for c in cajas
        ]
