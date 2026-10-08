"""Script principal para analizar imágenes con Ollama y Moondream.

Uso:
  python analizar_moondream.py ./prueba.jpg
  python analizar_moondream.py ./prueba.jpg --prompt "Describe la imagen y di si la persona tiene los ojos cerrados"
  python analizar_moondream.py --capturar          # Captura foto desde la webcam y la analiza
  python analizar_moondream.py --carpeta salida/capturas  # Analiza todas las capturas
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from src import config
from src.camara import Camara
from src.ia_moondream import ClienteMoondream, detectar_ollama_host


def imprimir_banner():
    print("=" * 65)
    print("   ANALIZADOR VISUAL CON MOONDREAM + OLLAMA")
    print("   Compatible con Windows y Raspberry Pi")
    print("=" * 65)


def analizar_archivo(cliente: ClienteMoondream, ruta_imagen: Path, prompt: str):
    print(f"\n[Analizando] {ruta_imagen.name}")
    print(f"[Prompt]    \"{prompt}\"")
    print(f"[Servidor]  {cliente.host}")
    print("Procesando imagen con Moondream...")

    res = cliente.consultar(ruta_imagen, prompt=prompt)

    if not res["ok"]:
        print(f"\n[ERROR] Falló la inferencia: {res['error']}")
        return res

    print("\n" + "-" * 55)
    print("DESCRIPCIÓN DE LA IA:")
    print(res["texto"])
    print("-" * 55)

    ojos = res["ojos_cerrados"]
    if ojos is True:
        print("DIAGNÓSTICO OJOS: [!] OJOS CERRADOS DETECTADOS (Posible somnolencia / descanso)")
    elif ojos is False:
        print("DIAGNÓSTICO OJOS: [OK] OJOS ABIERTOS")
    else:
        print("DIAGNÓSTICO OJOS: [-] No concluyente o no se enfoca en el estado de los ojos")

    print(f"Tiempo de respuesta: {res['duracion_seg']} s | Servidor: {res['servidor']}")
    print("=" * 65)
    return res


def capturar_y_analizar(cliente: ClienteMoondream, prompt: str, indice_cam: int):
    print(f"\nAbriendo webcam (índice {indice_cam})...")
    try:
        with Camara(indice_cam, config.ANCHO, config.ALTO, config.FPS, config.ESPEJO) as cam:
            print("Cámara lista. Tomando captura...")
            time.sleep(0.5)
            frame = cam.leer()
            if frame is None:
                print("Error: No se pudo obtener imagen de la cámara.")
                return

            ruta_salida = config.CARPETA_CAPTURAS / f"captura_manual_{int(time.time())}.jpg"
            config.CARPETA_CAPTURAS.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ruta_salida), frame)
            print(f"Captura guardada en: {ruta_salida}")
            analizar_archivo(cliente, ruta_salida, prompt)
    except Exception as e:
        print(f"Error al acceder a la cámara: {e}")


def main():
    p = argparse.ArgumentParser(description="Analizar imágenes con Moondream y Ollama")
    p.add_argument("imagen", nargs="?", default="prueba.jpg", help="Ruta a la imagen (por defecto ./prueba.jpg)")
    p.add_argument("--prompt", "-p", default=config.OLLAMA_PROMPT_DEFECTO, help="Pregunta o instrucción para Moondream")
    p.add_argument("--capturar", "-c", action="store_true", help="Tomar una foto con la webcam y analizarla")
    p.add_argument("--camara", type=int, default=config.INDICE_CAMARA, help="Índice de la webcam si se usa --capturar")
    p.add_argument("--host", default=None, help="URL de Ollama (ej. http://localhost:11434 o http://192.168.79.28:11434)")
    p.add_argument("--modelo", default=config.OLLAMA_MODELO, help="Nombre del modelo (por defecto 'moondream')")
    p.add_argument("--carpeta", help="Analizar todas las imágenes .jpg de una carpeta")
    args = p.parse_args()

    imprimir_banner()

    cliente = ClienteMoondream(host=args.host, modelo=args.modelo)
    disponible, msg = cliente.verificar_disponibilidad()
    print(f"Estado del servicio: {msg}")

    if not disponible:
        print("\nADVERTENCIA: Ollama no respondió en la dirección configurada.")
        print("Asegúrate de que Ollama está activo corriendo: ollama serve")
        print(f"Si usas la Raspberry Pi desde Windows, verifica que esté conectada en {detectar_ollama_host()}.\n")

    if args.capturar:
        capturar_y_analizar(cliente, args.prompt, args.camara)
        return

    if args.carpeta:
        carpeta = Path(args.carpeta)
        if not carpeta.is_dir():
            print(f"La carpeta '{args.carpeta}' no existe.")
            sys.exit(1)
        imagenes = sorted(carpeta.glob("*.jpg")) + sorted(carpeta.glob("*.png"))
        print(f"Se encontraron {len(imagenes)} imágenes en {carpeta}.")
        for img in imagenes:
            analizar_archivo(cliente, img, args.prompt)
        return

    ruta_img = Path(args.imagen)
    if not ruta_img.exists():
        print(f"El archivo '{ruta_img}' no existe.")
        print("Puedes pasar una ruta válida, o usar '--capturar' para tomar una foto con la webcam.")
        sys.exit(1)

    analizar_archivo(cliente, ruta_img, args.prompt)


if __name__ == "__main__":
    main()
