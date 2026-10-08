"""Dibujo sobre el vídeo. (cv2.putText no admite tildes, por eso los textos van sin ellas.)"""
import math
import time

import cv2
import numpy as np

from .analisis import geometria as g
from .identificadores import Deteccion

VERDE, AMARILLO, ROJO = (0, 220, 0), (0, 220, 255), (0, 0, 255)
GRIS, BLANCO, CIAN, NARANJA = (160, 160, 160), (255, 255, 255), (255, 220, 0), (0, 140, 255)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
COLOR_OJOS = {"abiertos": VERDE, "entrecerrados": AMARILLO, "cerrados": ROJO}
ETIQUETAS_RADAR = {"ojo_a": "ojo", "ojo_b": "ojo", "ceja_a": "ceja", "ceja_b": "ceja",
                   "labio_sup": "lab.sup", "labio_inf": "lab.inf", "comisura_a": "comis",
                   "comisura_b": "comis", "barbilla": "barbilla"}


def _texto(img, txt, xy, color=BLANCO, escala=0.55, grosor=1):
    cv2.putText(img, txt, xy, FUENTE, escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, FUENTE, escala, color, grosor, cv2.LINE_AA)


def _color_desviacion(rel: float):
    d = abs(rel - 1)
    return VERDE if d < 0.05 else AMARILLO if d < 0.12 else ROJO


class Dibujante:
    def __init__(self):
        self._eventos: list[tuple[float, str]] = []  # (hasta_cuando, texto) para mostrarlos un rato

    def dibujar(self, frame, detecciones: list[Deteccion], fps: float):
        ahora = time.monotonic()
        for d in detecciones:
            for e in d.extra.get("eventos", []):
                self._eventos.append((ahora + 2.0, e["tipo"].replace("_", " ").upper()))
            if "puntos" in d.privado:
                self._gestos(frame, d)
            elif d.etiqueta != "sin_rostro":
                x, y, w, h = d.caja
                color = ROJO if d.etiqueta == "ojos_cerrados" else VERDE
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                _texto(frame, f"{d.etiqueta} {d.confianza:.2f}", (x, max(y - 8, 15)), color, 0.6, 2)
                if "texto_ia" in d.privado:
                    for idx_t, linea in enumerate(d.privado["texto_ia"]):
                        c_lin = ROJO if "ALERTA" in linea else (VERDE if "OK" in linea else CIAN)
                        _texto(frame, linea, (20, 35 + idx_t * 26), c_lin, 0.6, 1)

        self._eventos = [(t, s) for t, s in self._eventos if t > ahora][-4:]
        for i, (_, s) in enumerate(reversed(self._eventos)):
            _texto(frame, s, (frame.shape[1] // 2 - 120, 40 + 32 * i), NARANJA, 0.8, 2)

        _texto(frame, f"FPS {fps:.1f} | q: salir  c: captura  r: recalibrar",
               (10, frame.shape[0] - 12), CIAN, 0.5)
        return frame

    # ---------------- modo gestos ----------------
    def _gestos(self, img, d: Deteccion):
        p, info = d.privado["puntos"], d.extra
        ojos, boca = info["ojos"], info["boca"]

        # Contornos de ojos y boca
        c_ojos = COLOR_OJOS.get(ojos["estado"], GRIS)
        for idx in (g.CONTORNO_OJO_A, g.CONTORNO_OJO_B):
            cv2.polylines(img, [p[idx, :2].astype(np.int32)], True, c_ojos, 1, cv2.LINE_AA)
        c_boca = NARANJA if boca.get("bostezando") else (200, 200, 200)
        cv2.polylines(img, [p[g.BOCA_INTERIOR, :2].astype(np.int32)], True, c_boca, 1, cv2.LINE_AA)

        # Punto radial y sus radios (color = cuánto se desvía de la base de la persona)
        ancla = tuple(p[g.ANCLA, :2].astype(int))
        radial = info.get("radial", {})
        for nombre, i in g.RADIOS.items():
            color = _color_desviacion(radial[nombre]) if nombre in radial else GRIS
            cv2.line(img, ancla, tuple(p[i, :2].astype(int)), color, 1, cv2.LINE_AA)
            cv2.circle(img, tuple(p[i, :2].astype(int)), 3, color, -1, cv2.LINE_AA)
        cv2.circle(img, ancla, 5, CIAN, -1, cv2.LINE_AA)

        self._panel(img, info)
        if radial:
            self._radar(img, radial)

    def _panel(self, img, info):
        ojos, boca, cont, ang = info["ojos"], info["boca"], info["contadores"], info["angulos"]
        x, y, dy = 10, 25, 24
        cv2.rectangle(img, (0, 0), (330, 10 + dy * 9), (0, 0, 0), -1)
        _texto(img, f"Orientacion: {info['orientacion'].upper()}", (x, y), CIAN, 0.6, 2)
        _texto(img, f"giro {ang['giro']:+.0f}  incl {ang['inclinacion']:+.0f}  ladeo {ang['ladeo']:+.0f}",
               (x, y + dy))
        if not info["calibrado"]:
            if ojos["estado"] == "no_fiable":
                _texto(img, "Perfil extremo: gire hacia la camara", (x, y + 2 * dy), AMARILLO)
            else:
                pr = info["calibracion_progreso"]
                _texto(img, f"Calibrando {info['orientacion']}: {pr:.0%}", (x, y + 2 * dy), AMARILLO)
                _texto(img, "Mire normal, ojos abiertos, boca cerrada", (x, y + 3 * dy), AMARILLO, 0.45)
                cv2.rectangle(img, (x, y + 3 * dy + 10), (x + int(300 * pr), y + 3 * dy + 18), AMARILLO, -1)
            return
        c = COLOR_OJOS.get(ojos["estado"], GRIS)
        _texto(img, f"Ojos: {ojos['estado']}  ({ojos['apertura_rel']:.0%} de lo normal)", (x, y + 2 * dy), c)
        _texto(img, f"EAR {ojos['ear']:.3f} / base {ojos['ear_base']:.3f}  [{ojos['ojo_usado']}]",
               (x, y + 3 * dy), GRIS, 0.45)
        _texto(img, f"Boca MAR {boca['mar']:.2f} / base {boca['mar_base']:.2f}"
               + ("  BOSTEZO" if boca["bostezando"] else ""), (x, y + 4 * dy),
               NARANJA if boca["bostezando"] else BLANCO)
        _texto(img, f"Parpadeos {cont['parpadeos']}  ({cont['parpadeos_por_min']:.0f}/min)", (x, y + 5 * dy))
        _texto(img, f"Bostezos {cont['bostezos']}", (x, y + 6 * dy))
        perclos = info["perclos"]
        _texto(img, f"PERCLOS {perclos:.0%} (ojos cerrados, ult. min)", (x, y + 7 * dy),
               ROJO if perclos > 0.15 else BLANCO)

    def _radar(self, img, radial: dict[str, float]):
        """Gráfico radial: circulo = base de la persona (100%); poligono = ahora mismo."""
        r_max, lado = 95, 230
        cx, cy = img.shape[1] - lado // 2 - 10, lado // 2 + 10
        x0, y0 = cx - lado // 2, cy - lado // 2
        sub = img[y0:y0 + lado, x0:x0 + lado]
        sub[:] = (sub * 0.35).astype(np.uint8)
        r_base = int(r_max * 0.6)
        cv2.circle(img, (cx, cy), r_base, CIAN, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r_max, GRIS, 1, cv2.LINE_AA)

        nombres = list(radial)
        pts = []
        for k, nombre in enumerate(nombres):
            ang = -math.pi / 2 + 2 * math.pi * k / len(nombres)
            ux, uy = math.cos(ang), math.sin(ang)
            cv2.line(img, (cx, cy), (int(cx + ux * r_max), int(cy + uy * r_max)), (90, 90, 90), 1)
            # 100% -> circulo base; cada 10% de diferencia se ve como 20% del radio
            r = r_max * min(max(0.6 + (radial[nombre] - 1) * 2.0, 0.05), 1.0)
            pts.append((int(cx + ux * r), int(cy + uy * r)))
            _texto(img, ETIQUETAS_RADAR.get(nombre, nombre),
                   (int(cx + ux * (r_max + 4)) - 18, int(cy + uy * (r_max + 4)) + 4), GRIS, 0.35)
        cv2.polylines(img, [np.array(pts, np.int32)], True, VERDE, 2, cv2.LINE_AA)
        for (px, py), nombre in zip(pts, nombres):
            cv2.circle(img, (px, py), 3, _color_desviacion(radial[nombre]), -1, cv2.LINE_AA)
        _texto(img, "Radial vs base", (x0 + 5, y0 + lado - 6), CIAN, 0.45)
