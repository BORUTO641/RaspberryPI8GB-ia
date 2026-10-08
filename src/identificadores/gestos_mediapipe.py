"""Identificador de gestos faciales con MediaPipe Face Landmarker (478 puntos, local, CPU)."""
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .. import config
from ..analisis import geometria as g
from ..analisis.gestos import AnalizadorGestos
from .base import Deteccion, Identificador


class IdentificadorGestos(Identificador):
    nombre = "gestos_mediapipe"

    def __init__(self):
        if not config.RUTA_MODELO_ROSTRO.exists():
            raise FileNotFoundError(
                f"Falta el modelo {config.RUTA_MODELO_ROSTRO}. Descárgalo de "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )
        opciones = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.RUTA_MODELO_ROSTRO)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opciones)
        self.analizador = AnalizadorGestos()
        self._t0 = time.monotonic()
        self._ultimo_ms = -1

    def identificar(self, frame) -> list[Deteccion]:
        t = time.monotonic() - self._t0
        ms = max(int(t * 1000), self._ultimo_ms + 1)  # MediaPipe exige tiempos crecientes
        self._ultimo_ms = ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)

        if not res.face_landmarks:
            eventos = self.analizador.sin_rostro(t)
            # Detección "vacía" solo para transportar el evento de rostro perdido
            return [Deteccion("sin_rostro", 0.0, (0, 0, 0, 0), {"eventos": eventos})] if eventos else []

        alto, ancho = frame.shape[:2]
        p = g.puntos_3d(res.face_landmarks[0], ancho, alto)
        info = self.analizador.procesar(p, t)
        return [Deteccion("rostro", 1.0, g.caja(p), info, privado={"puntos": p})]

    def recalibrar(self):
        self.analizador = AnalizadorGestos()

    def cerrar(self):
        self.landmarker.close()
