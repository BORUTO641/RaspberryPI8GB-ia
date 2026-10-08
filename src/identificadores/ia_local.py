"""Identificador multimodal local usando Ollama + Moondream.

Funciona de forma asíncrona mediante un hilo en segundo plano:
- Mantiene 30 FPS en la cámara sin congelar la ventana.
- Consulta Moondream cada N segundos para describir la escena y detectar si los ojos están cerrados.
- Integra seguimiento facial en tiempo real para el recuadro.
"""
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from .. import config
from ..ia_moondream import ClienteMoondream
from .base import Deteccion, Identificador


def _buscar_haarcascade() -> Optional[cv2.CascadeClassifier]:
    rutas_candidatas = []
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        rutas_candidatas.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    rutas_candidatas.extend([
        Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
        Path("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
    ])
    try:
        cv2_dir = Path(cv2.__file__).resolve().parent
        rutas_candidatas.extend(cv2_dir.glob("**/haarcascade_frontalface_default.xml"))
    except Exception:
        pass

    for r in rutas_candidatas:
        if r.is_file():
            clf = cv2.CascadeClassifier(str(r))
            if not clf.empty():
                return clf
    return None


class IdentificadorIALocal(Identificador):
    nombre = "ia_local"

    def __init__(self):
        self.cliente = ClienteMoondream(
            host=config.OLLAMA_HOST,
            modelo=config.OLLAMA_MODELO,
            timeout=30.0
        )
        ok, msg = self.cliente.verificar_disponibilidad()
        print(f"[{self.nombre}] {msg}")

        # Detector de rostros Haar ligero para ubicar el recuadro a 30 FPS (opcional)
        self._cascade = _buscar_haarcascade()

        self._cola_frames: queue.Queue = queue.Queue(maxsize=1)
        self._activo = True
        self._ultimo_envio = 0.0
        self._intervalo = config.OLLAMA_INTERVALO_SEG

        # Últimos resultados de la inferencia
        self._lock = threading.Lock()
        self._ultima_respuesta = "Iniciando analisis de vision..."
        self._ojos_cerrados: Optional[bool] = None
        self._tiempo_inferencia = 0.0
        self._nuevo_evento = False

        # Iniciar hilo de inferencia asíncrono
        self._hilo = threading.Thread(target=self._bucle_inferencia, daemon=True)
        self._hilo.start()

    def _bucle_inferencia(self):
        while self._activo:
            try:
                frame = self._cola_frames.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                res = self.cliente.consultar(
                    frame,
                    prompt=config.OLLAMA_PROMPT_DEFECTO,
                    temperatura=0.2,
                    max_tokens=120
                )
                with self._lock:
                    if res["ok"]:
                        self._ultima_respuesta = res["texto"]
                        self._ojos_cerrados = res["ojos_cerrados"]
                        self._tiempo_inferencia = res["duracion_seg"]
                        if res["ojos_cerrados"] is True:
                            self._nuevo_evento = True
                    else:
                        self._ultima_respuesta = f"Error: {res['error']}"
            except Exception as e:
                with self._lock:
                    self._ultima_respuesta = f"Excepcion: {e}"
            finally:
                self._cola_frames.task_done()

    def identificar(self, frame) -> list[Deteccion]:
        ahora = time.monotonic()
        alto, ancho = frame.shape[:2]

        # Enviar frame al modelo cada N segundos si el hilo está libre
        if ahora - self._ultimo_envio >= self._intervalo:
            if self._cola_frames.empty():
                try:
                    self._cola_frames.put_nowait(frame.copy())
                    self._ultimo_envio = ahora
                except queue.Full:
                    pass

        caja = (int(ancho * 0.25), int(alto * 0.15), int(ancho * 0.5), int(alto * 0.7))
        if self._cascade is not None:
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostros = self._cascade.detectMultiScale(gris, 1.2, 4, minSize=(60, 60))
            if len(rostros) > 0:
                caja = tuple(int(v) for v in rostros[0])

        with self._lock:
            resp_texto = self._ultima_respuesta
            ojos = self._ojos_cerrados
            t_inf = self._tiempo_inferencia
            hay_alerta = self._nuevo_evento
            self._nuevo_evento = False

        if ojos is True:
            etiqueta = "ojos_cerrados"
        elif ojos is False:
            etiqueta = "ojos_abiertos"
        else:
            etiqueta = "moondream_ia"

        eventos = []
        if hay_alerta:
            eventos.append({"tipo": "ojos_cerrados_alerta", "capturar": True})

        extra = {
            "modelo": config.OLLAMA_MODELO,
            "descripcion": resp_texto,
            "ojos_cerrados": ojos,
            "tiempo_inferencia_seg": t_inf,
            "eventos": eventos
        }

        # Preparar texto en líneas cortas para visualización
        lineas = []
        if ojos is True:
            lineas.append("[ALERTA] OJOS CERRADOS DETECTADOS")
        elif ojos is False:
            lineas.append("[OK] OJOS ABIERTOS")
        
        if t_inf > 0:
            lineas.append(f"Moondream: {t_inf:.1f}s | {self.cliente.host}")
            
        # Resumen del texto
        palabras = resp_texto.split()
        if palabras:
            lineas.append(" ".join(palabras[:8]) + ("..." if len(palabras) > 8 else ""))

        privado = {"texto_ia": lineas}

        return [Deteccion(etiqueta=etiqueta, confianza=0.95, caja=caja, extra=extra, privado=privado)]

    def recalibrar(self):
        with self._lock:
            self._ultima_respuesta = "Recalibrando Moondream..."
            self._ojos_cerrados = None
        self._ultimo_envio = 0.0

    def cerrar(self):
        self._activo = False
        try:
            while not self._cola_frames.empty():
                self._cola_frames.get_nowait()
                self._cola_frames.task_done()
        except Exception:
            pass
        if self._hilo.is_alive():
            self._hilo.join(timeout=1.0)
