"""Punto de entrada:  python main.py   |   python main.py --listar"""
import argparse
import os
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # oculta avisos al buscar cámaras inexistentes

import cv2

from src import config
from src.camara import Camara, listar_camaras
from src.dibujo import Dibujante
from src.identificadores import MODOS, crear_identificador
from src.salida import RegistroSalida


def main():
    p = argparse.ArgumentParser(description="Identificación con webcam USB")
    p.add_argument("--listar", action="store_true", help="lista las cámaras disponibles")
    p.add_argument("--camara", default=config.INDICE_CAMARA, help="índice de cámara (0, 1) o ruta de imagen/vídeo")
    p.add_argument("--modo", default=config.MODO_IDENTIFICADOR, choices=MODOS)
    p.add_argument("--sin-ventana", action="store_true", help="no mostrar vídeo (solo registrar)")
    p.add_argument("--segundos", type=float, default=0, help="terminar tras N segundos (0 = sin límite)")
    args = p.parse_args()

    if args.listar:
        camaras = listar_camaras()
        if not camaras:
            print("No se encontró ninguna cámara.")
        for indice, nombre, estado in camaras:
            print(f"  [{indice}] {nombre:<25} {estado}")
        return

    identificador = crear_identificador(args.modo)
    registro = RegistroSalida(config.ARCHIVO_DETECCIONES, config.CARPETA_CAPTURAS,
                              config.GUARDAR_CAPTURA_AL_DETECTAR, config.SEGUNDOS_ENTRE_CAPTURAS,
                              config.INTERVALO_REGISTRO_SEG)
    dibujante = Dibujante()
    n_analisis = config.ANALIZAR_CADA_N_FRAMES if args.modo == "rostros" else 1

    with Camara(args.camara, config.ANCHO, config.ALTO, config.FPS, config.ESPEJO) as cam:
        print(f"Cámara {args.camara} abierta a {cam.resolucion()} | identificador: {identificador.nombre}")
        print(f"Resultados en: {config.ARCHIVO_DETECCIONES}")
        n, detecciones, fps = 0, [], 0.0
        inicio = t0 = time.perf_counter()
        try:
            while not args.segundos or time.perf_counter() - inicio < args.segundos:
                frame = cam.leer()
                if frame is None:
                    print("No llegan imágenes de la cámara.")
                    break

                if n % n_analisis == 0:
                    detecciones = identificador.identificar(frame)
                    registro.registrar(frame, detecciones, identificador.nombre)
                    for d in detecciones:
                        for e in d.extra.get("eventos", []):
                            print(f"[{time.strftime('%H:%M:%S')}] {e['tipo']}"
                                  + (f"  {e['duracion']} s" if "duracion" in e else ""))
                n += 1

                t1 = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1 / max(t1 - t0, 1e-6))
                t0 = t1

                if args.sin_ventana:
                    continue
                cv2.imshow("Identificacion webcam", dibujante.dibujar(frame.copy(), detecciones, fps))
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("c"):
                    registro.registrar(frame, detecciones, identificador.nombre, forzar_captura=True)
                    print("Captura guardada.")
                if tecla == ord("r"):
                    identificador.recalibrar()
                    print("Recalibrando: mire a la cámara con normalidad.")
        except KeyboardInterrupt:
            pass
        finally:
            identificador.cerrar()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
