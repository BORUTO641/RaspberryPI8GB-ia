"""Escribe los resultados en disco en un formato fácil de leer para la IA u otro programa.

- salida/detecciones.jsonl : una línea JSON por evento (parpadeo, bostezo...) al instante,
                             y un estado periódico cada INTERVALO_REGISTRO_SEG.
- salida/capturas/*.jpg    : imagen de los eventos importantes (o de cada detección si el
                             identificador no genera eventos, como el modo "rostros").
"""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from .identificadores import Deteccion


class RegistroSalida:
    def __init__(self, archivo: Path, carpeta_capturas: Path, guardar_capturas: bool,
                 segundos_entre_capturas: float, intervalo_registro: float):
        archivo.parent.mkdir(parents=True, exist_ok=True)
        carpeta_capturas.mkdir(parents=True, exist_ok=True)
        self.archivo = archivo
        self.carpeta_capturas = carpeta_capturas
        self.guardar_capturas = guardar_capturas
        self.segundos_entre_capturas = segundos_entre_capturas
        self.intervalo_registro = intervalo_registro
        self._ultima_captura = 0.0
        self._ultimo_registro = 0.0

    def registrar(self, frame, detecciones: list[Deteccion], fuente: str,
                  forzar_captura: bool = False):
        ahora_m = time.monotonic()
        usa_eventos = any("eventos" in d.extra for d in detecciones)
        eventos = [e for d in detecciones for e in d.extra.get("eventos", [])]
        periodico = bool(detecciones) and ahora_m - self._ultimo_registro >= self.intervalo_registro
        if not (eventos or periodico or forzar_captura):
            return

        quiere_captura = any(e.get("capturar") for e in eventos) if usa_eventos else bool(detecciones)
        ruta_img = None
        if forzar_captura or (
            self.guardar_capturas and quiere_captura
            and ahora_m - self._ultima_captura >= self.segundos_entre_capturas
        ):
            ruta_img = self.carpeta_capturas / f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(str(ruta_img), frame)
            self._ultima_captura = ahora_m

        registro = {
            "fecha": datetime.now().isoformat(timespec="milliseconds"),
            "fuente": fuente,
            "tipo": "evento" if eventos else "estado",
            "eventos": [e["tipo"] for e in eventos],
            "ancho": frame.shape[1],
            "alto": frame.shape[0],
            "imagen": ruta_img.relative_to(self.archivo.parent).as_posix() if ruta_img else None,
            "detecciones": [d.a_json() for d in detecciones],
        }
        with self.archivo.open("a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        self._ultimo_registro = ahora_m
