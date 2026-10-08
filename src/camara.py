"""Acceso a la webcam USB (Windows y Linux / Raspberry Pi)."""
import platform
from pathlib import Path

import cv2

SISTEMA = platform.system()
# Windows: DirectShow abre la cámara mucho más rápido que MSMF. Linux: V4L2.
BACKEND = {"Windows": cv2.CAP_DSHOW, "Linux": cv2.CAP_V4L2}.get(SISTEMA, cv2.CAP_ANY)


def nombres_camaras() -> list[tuple[int, str]]:
    """(índice, nombre) de cada cámara del sistema."""
    if SISTEMA == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
            return list(enumerate(FilterGraph().get_input_devices()))
        except Exception:
            return []
    if SISTEMA == "Linux":
        camaras = []
        for d in sorted(Path("/sys/class/video4linux").glob("video*"),
                        key=lambda p: int(p.name[5:])):
            try:
                # cada webcam crea 2 nodos; el de índice 0 es el de captura
                if (d / "index").read_text().strip() == "0":
                    camaras.append((int(d.name[5:]), (d / "name").read_text().strip()))
            except OSError:
                pass
        return camaras
    return []


def buscar_camara(camara: int | str):
    """Acepta un índice (1), ruta de vídeo/imagen o parte del nombre ("Webcam")."""
    if isinstance(camara, int) or str(camara).isdigit():
        return int(camara)
    p = Path(str(camara))
    if p.is_file():
        return str(p)
    for indice, nombre in nombres_camaras():
        if str(camara).lower() in nombre.lower():
            return indice
    raise RuntimeError(f"No hay ninguna cámara ni archivo llamado {camara!r}. Usa `python main.py --listar`.")


def listar_camaras() -> list[tuple[int, str, str]]:
    """Devuelve (índice, nombre, estado) de cada cámara encontrada."""
    nombres = nombres_camaras() or [(i, "?") for i in range(5)]
    resultado = []
    for i, nombre in nombres:
        cap = cv2.VideoCapture(i, BACKEND)
        if cap.isOpened():
            estado = "OK" if cap.read()[0] else "sin imagen (¿la usa otra app?)"
            resultado.append((i, nombre, estado))
        elif nombre != "?":
            resultado.append((i, nombre, "no se puede abrir (¿la usa otra app?)"))
        cap.release()
    return resultado


class Camara:
    def __init__(self, camara: int | str, ancho: int, alto: int, fps: int, espejo: bool = False):
        self.espejo = espejo
        self.origen = buscar_camara(camara)
        self._es_archivo = isinstance(self.origen, str) and Path(self.origen).is_file()
        self._img_estatica = None

        if self._es_archivo and Path(self.origen).suffix.lower() in (".jpg", ".jpeg", ".png"):
            self._img_estatica = cv2.imread(self.origen)
            if self._img_estatica is None:
                raise RuntimeError(f"No se pudo cargar la imagen {self.origen!r}")
            self.cap = None
        else:
            backend = cv2.CAP_ANY if self._es_archivo else BACKEND
            self.cap = cv2.VideoCapture(self.origen, backend)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"No se pudo abrir la cámara o archivo {camara!r} (origen {self.origen}). "
                    "¿Está conectada? ¿La usa otra app? Prueba `python main.py --listar`."
                )
            if not self._es_archivo:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._calentar()

    def _calentar(self, frames: int = 15):
        """Descarta los primeros frames en cámaras reales para estabilizar exposición."""
        if self._es_archivo:
            return
        frame = None
        for _ in range(frames):
            frame = self.leer()
        if frame is None:
            self.cerrar()
            raise RuntimeError(
                "La cámara se abrió pero no envía imágenes. Casi siempre es porque otra "
                "aplicación la está usando (navegador, Discord, Teams, Zoom, OBS, app Cámara). "
                "Ciérrala y vuelve a intentar."
            )
        if frame.mean() < 10:
            print("AVISO: la imagen está casi negra. ¿Tapa de privacidad cerrada, lente tapada "
                  "o es otra cámara? Prueba otra con --camara N.")

    def leer(self):
        if self._img_estatica is not None:
            import time
            time.sleep(0.03)  # simula ~30 FPS
            frame = self._img_estatica.copy()
            return cv2.flip(frame, 1) if self.espejo else frame

        if self.cap is None:
            return None

        ok, frame = self.cap.read()
        if not ok:
            if self._es_archivo:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None
        return cv2.flip(frame, 1) if self.espejo else frame

    def resolucion(self) -> tuple[int, int]:
        if self._img_estatica is not None:
            return (self._img_estatica.shape[1], self._img_estatica.shape[0])
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def cerrar(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cerrar()
