"""Medidas geométricas sobre los 478 puntos de MediaPipe Face Landmarker.

Todas las distancias se calculan en 3D y se normalizan por la distancia entre ojos,
así no dependen de lo lejos que esté la persona ni del tamaño de su cara.
"""
import math

import numpy as np

# Índices de la malla facial de MediaPipe
OJO_A = [33, 160, 158, 133, 153, 144]     # p1..p6 para EAR (un ojo)
OJO_B = [362, 385, 387, 263, 373, 380]    # p1..p6 para EAR (el otro ojo)
CONTORNO_OJO_A = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
CONTORNO_OJO_B = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
BOCA_INTERIOR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
BOCA_VERTICALES = [(82, 87), (13, 14), (312, 317)]
BOCA_COMISURAS = (78, 308)
ESQUINAS_EXTERIORES = (33, 263)           # para normalizar (distancia entre ojos)
MEJILLAS = (234, 454)                     # para el giro izquierda/derecha
FRENTE_BARBILLA = (10, 152)               # para el giro arriba/abajo

# Punto radial (ancla) y los radios que se comparan en tiempo real
ANCLA = 168  # entrecejo: casi no se mueve al gesticular
RADIOS = {
    "ojo_a": 468,        # centro del iris
    "ojo_b": 473,
    "ceja_a": 105,
    "ceja_b": 334,
    "labio_sup": 13,
    "labio_inf": 14,
    "comisura_a": 61,
    "comisura_b": 291,
    "barbilla": 152,
}


def puntos_3d(landmarks, ancho: int, alto: int) -> np.ndarray:
    """Convierte los landmarks normalizados a píxeles (x, y) y profundidad z en la misma escala."""
    return np.array([(p.x * ancho, p.y * alto, p.z * ancho) for p in landmarks], dtype=np.float32)


def _d(p, a, b) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


def ear(p, idx) -> float:
    """Eye Aspect Ratio: alto del ojo / ancho del ojo. ~0.25-0.35 abierto, <0.15 cerrado
    (varía según la persona; por eso luego se compara con su propia base)."""
    p1, p2, p3, p4, p5, p6 = idx
    return (_d(p, p2, p6) + _d(p, p3, p5)) / (2.0 * _d(p, p1, p4) + 1e-6)


def mar(p) -> float:
    """Mouth Aspect Ratio: apertura interior de la boca / ancho de la boca."""
    alto = sum(_d(p, a, b) for a, b in BOCA_VERTICALES) / len(BOCA_VERTICALES)
    return alto / (_d(p, *BOCA_COMISURAS) + 1e-6)


def profundidad_ojo(p, idx) -> float:
    """z medio del ojo: el menor es el ojo más cercano a la cámara (el que se ve mejor de lado)."""
    return float(np.mean(p[idx, 2]))


def orientacion(p) -> tuple[float, float, float]:
    """Ángulos aproximados de la cabeza en grados: (giro lateral, inclinación vertical, ladeo)."""
    v = p[MEJILLAS[1]] - p[MEJILLAS[0]]
    u = p[FRENTE_BARBILLA[1]] - p[FRENTE_BARBILLA[0]]
    giro = math.degrees(math.atan2(v[2], v[0]))
    inclinacion = math.degrees(math.atan2(u[2], u[1]))
    ladeo = math.degrees(math.atan2(v[1], v[0]))
    return giro, inclinacion, ladeo


def radios(p) -> dict[str, float]:
    """Distancia del punto radial (entrecejo) a cada punto clave, en unidades de 'distancia entre ojos'."""
    escala = _d(p, *ESQUINAS_EXTERIORES) + 1e-6
    return {nombre: _d(p, ANCLA, i) / escala for nombre, i in RADIOS.items()}


def caja(p) -> tuple[int, int, int, int]:
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    x1, y1 = p[:, 0].max(), p[:, 1].max()
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
