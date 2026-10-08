"""Consumidor de eventos en tiempo real con integración de IA multimodal (Moondream).

Sigue el archivo detecciones.jsonl (como `tail -f`) e imprime cada evento nuevo.
Si el evento incluye una captura de imagen, la analiza con Moondream.
"""
import argparse
import json
import time
from pathlib import Path

from src import config
from src.ia_moondream import ClienteMoondream


def seguir(ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.touch(exist_ok=True)
    with ruta.open(encoding="utf-8") as f:
        f.seek(0, 2)  # empezar por el final: solo eventos nuevos
        while True:
            linea = f.readline()
            if not linea:
                time.sleep(0.2)
                continue
            try:
                yield json.loads(linea)
            except json.JSONDecodeError:
                continue


def main():
    p = argparse.ArgumentParser(description="Escucha detecciones y analiza eventos con Moondream")
    p.add_argument("--ia", action="store_true", default=True, help="Analizar imágenes con Moondream (por defecto activado)")
    p.add_argument("--sin-ia", dest="ia", action="store_false", help="Solo mostrar eventos sin invocar la IA")
    p.add_argument("--host", default=None, help="URL de Ollama")
    args = p.parse_args()

    cliente = None
    if args.ia:
        cliente = ClienteMoondream(host=args.host)
        ok, msg = cliente.verificar_disponibilidad()
        print(f"[Moondream] {msg}")

    print(f"Escuchando {config.ARCHIVO_DETECCIONES} ... (Ctrl+C para salir)\n")
    try:
        for evento in seguir(config.ARCHIVO_DETECCIONES):
            fecha = evento.get("fecha", "")
            eventos = evento.get("eventos", [])
            img_rel = evento.get("imagen")
            fuente = evento.get("fuente", "")

            txt_eventos = f" | Eventos: {', '.join(eventos)}" if eventos else ""
            print(f"[{fecha}] [{fuente}]{txt_eventos}")

            if img_rel and args.ia and cliente:
                ruta_img = config.CARPETA_SALIDA / img_rel
                if ruta_img.is_file():
                    print(f"  -> Analizando captura con Moondream: {ruta_img.name} ...")
                    res = cliente.consultar(
                        ruta_img,
                        prompt="Describe la imagen y responde si la persona tiene los ojos cerrados o abiertos."
                    )
                    if res["ok"]:
                        ojos = res["ojos_cerrados"]
                        diag = "[!] OJOS CERRADOS" if ojos is True else ("[OK] OJOS ABIERTOS" if ojos is False else "[-]")
                        print(f"  [Moondream {res['duracion_seg']}s] {diag}")
                        print(f"  \"{res['texto']}\"\n")
                    else:
                        print(f"  [Error Moondream] {res['error']}\n")
    except KeyboardInterrupt:
        print("\nFinalizado.")


if __name__ == "__main__":
    main()
