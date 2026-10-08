"""Módulo cliente para Ollama + Moondream (compatible con Windows y Raspberry Pi).

Permite enviar imágenes (archivos o frames numpy) a Moondream para:
- Describir la imagen.
- Detectar si la persona tiene los ojos cerrados / somnolencia.
- Responder preguntas visuales específicas.
"""
import base64
import json
import os
import platform
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2

# Host por defecto según entorno
DEFAULT_HOST_PI = "http://192.168.79.28:11434"
DEFAULT_HOST_LOCAL = "http://127.0.0.1:11434"


def puerto_abierto(host: str, puerto: int, timeout: float = 1.0) -> bool:
    """Comprueba rápidamente si un host y puerto responden."""
    try:
        with socket.create_connection((host, puerto), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def detectar_ollama_host() -> str:
    """Detecta automáticamente el endpoint de Ollama disponible.

    1. Variable de entorno OLLAMA_HOST si está definida.
    2. Localhost 127.0.0.1:11434 (siempre prioritario si está corriendo localmente).
    3. Si estamos en Windows y local no responde, intenta la IP de la Raspberry Pi.
    """
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        if not env_host.startswith("http://") and not env_host.startswith("https://"):
            env_host = f"http://{env_host}"
        return env_host

    # 1. Probar localhost
    if puerto_abierto("127.0.0.1", 11434, timeout=0.5):
        return DEFAULT_HOST_LOCAL

    # 2. Si estamos en Windows, probar Raspberry Pi en WiFi o Ethernet
    if platform.system() == "Windows":
        for ip in ("192.168.79.28", "192.168.4.10", "inteligenciaartificial.local"):
            if puerto_abierto(ip, 11434, timeout=0.8):
                return f"http://{ip}:11434"

    return DEFAULT_HOST_LOCAL


def codificar_imagen_b64(imagen: str | Path | Any, max_dim: int = 768) -> str:
    """Convierte una ruta de archivo o un frame BGR numpy a una cadena base64.

    Redimensiona proporcionalmente si supera max_dim para acelerar inferencia
    en dispositivos limitados como la Raspberry Pi.
    """
    if isinstance(imagen, (str, Path)):
        ruta = Path(imagen)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe la imagen: {ruta}")
        frame = cv2.imread(str(ruta))
        if frame is None:
            raise ValueError(f"No se pudo cargar la imagen desde: {ruta}")
    else:
        frame = imagen

    alto, ancho = frame.shape[:2]
    if max(alto, ancho) > max_dim:
        escala = max_dim / max(alto, ancho)
        nuevo_w = int(ancho * escala)
        nuevo_h = int(alto * escala)
        frame = cv2.resize(frame, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("No se pudo codificar el frame a JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def interpretar_ojos(texto: str) -> Optional[bool]:
    """Determina si la respuesta indica ojos cerrados (True), abiertos (False) o incierto (None)."""
    t = texto.lower()
    
    # Patrones para ojos cerrados / durmiendo
    patrones_cerrados = [
        r"ojos?\s+(est[aá]n\s+)?cerrados?",
        r"eyes?\s+(are\s+)?closed",
        r"has\s+his\s+eyes\s+closed",
        r"has\s+her\s+eyes\s+closed",
        r"eyes\s+shut",
        r"sleeping",
        r"dormid[oa]",
        r"asleep",
        r"resting\s+his\s+eyes"
    ]
    # Patrones para ojos abiertos
    patrones_abiertos = [
        r"ojos?\s+(est[aá]n\s+)?abiertos?",
        r"eyes?\s+(are\s+)?open",
        r"wide\s+open",
        r"awake",
        r"despiert[oa]"
    ]

    for p in patrones_cerrados:
        if re.search(p, t):
            return True
            
    for p in patrones_abiertos:
        if re.search(p, t):
            return False

    return None


class ClienteMoondream:
    """Cliente para interactuar con Moondream vía Ollama API o CLI."""

    def __init__(self, host: Optional[str] = None, modelo: str = "moondream", timeout: float = 60.0):
        self.modelo = modelo
        self.timeout = timeout
        self.host = host.rstrip("/") if host else detectar_ollama_host()

    def verificar_disponibilidad(self) -> tuple[bool, str]:
        """Comprueba si el servidor Ollama responde y si tiene moondream cargado."""
        url = f"{self.host}/api/tags"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "WebcamIdentificacion/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                datos = json.loads(resp.read().decode("utf-8"))
                modelos = [m.get("name", "").split(":")[0] for m in datos.get("models", [])]
                if self.modelo.split(":")[0] in modelos:
                    return True, f"Ollama activo en {self.host} con modelo '{self.modelo}' listo."
                return True, f"Ollama activo en {self.host}, pero '{self.modelo}' no está en la lista: {modelos}"
        except Exception as e:
            return False, f"No se pudo conectar a Ollama en {self.host}: {e}"

    def consultar(
        self,
        imagen: str | Path | Any,
        prompt: str = "Describe la imagen y di si la persona tiene los ojos cerrados",
        temperatura: float = 0.2,
        max_tokens: int = 150
    ) -> dict:
        """Envía la imagen a Moondream y devuelve un diccionario con el resultado."""
        t0 = time.perf_counter()
        img_b64 = codificar_imagen_b64(imagen)

        payload = {
            "model": self.modelo,
            "prompt": prompt,
            "images": [img_b64],
            "stream": True,
            "context": [],
            "options": {
                "temperature": temperatura,
                "num_predict": max_tokens,
                "repeat_penalty": 1.15
            }
        }

        url = f"{self.host}/api/generate"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            trozos = []
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for linea in resp:
                    if linea:
                        chunk = json.loads(linea.decode("utf-8"))
                        trozos.append(chunk.get("response", ""))
                        if chunk.get("done"):
                            break
            texto = "".join(trozos).strip()
            t1 = time.perf_counter()
            ojos_cerrados = interpretar_ojos(texto)
            return {
                "ok": True,
                "texto": texto,
                "ojos_cerrados": ojos_cerrados,
                "duracion_seg": round(t1 - t0, 2),
                "servidor": self.host,
                "modelo": self.modelo,
                "error": None
            }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "texto": "",
                "ojos_cerrados": None,
                "duracion_seg": round(time.perf_counter() - t0, 2),
                "servidor": self.host,
                "modelo": self.modelo,
                "error": f"HTTP {e.code}: {err_body}"
            }
        except Exception as e:
            # Fallback opcional: si falla la API y estamos en la máquina con CLI de Ollama, probar subprocess
            return self._consultar_cli_fallback(imagen, prompt, t0, str(e))

    def _consultar_cli_fallback(self, imagen: str | Path | Any, prompt: str, t0: float, err_original: str) -> dict:
        """Intento de respaldo usando el comando `ollama run moondream` del sistema."""
        try:
            # Si imagen es frame numpy o ruta temporal
            if not isinstance(imagen, (str, Path)) or not Path(imagen).is_file():
                temp_path = Path("/tmp/moondream_tmp.jpg" if platform.system() != "Windows" else "temp_moondream.jpg")
                if not isinstance(imagen, (str, Path)):
                    cv2.imwrite(str(temp_path), imagen)
                ruta_img = str(temp_path)
            else:
                ruta_img = str(Path(imagen).resolve())

            cmd = ["ollama", "run", self.modelo, f"{prompt}: {ruta_img}"]
            proceso = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if proceso.returncode == 0:
                texto = proceso.stdout.strip()
                return {
                    "ok": True,
                    "texto": texto,
                    "ojos_cerrados": interpretar_ojos(texto),
                    "duracion_seg": round(time.perf_counter() - t0, 2),
                    "servidor": "cli-local",
                    "modelo": self.modelo,
                    "error": None
                }
        except Exception:
            pass

        return {
            "ok": False,
            "texto": "",
            "ojos_cerrados": None,
            "duracion_seg": round(time.perf_counter() - t0, 2),
            "servidor": self.host,
            "modelo": self.modelo,
            "error": err_original
        }
