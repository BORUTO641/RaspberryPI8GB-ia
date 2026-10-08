"""Análisis de gestos faciales en tiempo real: ojos (parpadeo, entrecerrados, cerrados),
bostezo, orientación de la cabeza y comparación radial con la base de la persona.

Ideas clave:
- Cada persona tiene ojos y boca distintos -> no se usan umbrales fijos, sino una BASE
  calibrada de esa persona, y todo se mide en % respecto a esa base.
- La cabeza se mueve -> hay una base SEPARADA por orientación (frente, izquierda, derecha,
  arriba, abajo). De lado se usa el ojo que mira a la cámara.
- Estabilidad -> filtro One Euro en cada señal, histéresis en los umbrales y tiempos mínimos
  antes de cambiar de estado (evita parpadeos falsos por ruido o movimiento).
"""
from collections import deque

import numpy as np

from .. import config as C
from . import geometria as g
from .filtros import OneEuro

ORIENTACIONES = ["frente", "izquierda", "derecha", "arriba", "abajo"]


class BasePose:
    """Valores normales de la persona en UNA orientación de la cabeza."""

    def __init__(self):
        self._ear, self._mar, self._radios = [], [], []
        self.calibrada = False
        self.ear = self.mar = None
        self.radios: dict[str, float] = {}

    @property
    def progreso(self) -> float:
        return 1.0 if self.calibrada else len(self._ear) / C.CALIBRACION_MUESTRAS

    def agregar(self, ear: float, mar: float, radios: dict[str, float]):
        self._ear.append(ear)
        self._mar.append(mar)
        self._radios.append(radios)
        if len(self._ear) >= C.CALIBRACION_MUESTRAS:
            # percentil alto de EAR: ignora los parpadeos que ocurran durante la calibración
            self.ear = float(np.percentile(self._ear, 80))
            self.mar = float(np.percentile(self._mar, 30))
            self.radios = {k: float(np.median([r[k] for r in self._radios])) for k in radios}
            self.calibrada = True
            self._ear, self._mar, self._radios = [], [], []

    def adaptar(self, ear: float, mar: float, radios: dict[str, float]):
        """Ajuste lento de la base (luz, distancia, cansancio gradual)."""
        a = C.ADAPTACION_ALPHA
        self.ear = (1 - a) * self.ear + a * ear
        self.mar = (1 - a) * self.mar + a * mar
        for k, v in radios.items():
            self.radios[k] = (1 - a) * self.radios[k] + a * v


class AnalizadorGestos:
    def __init__(self):
        self.bases = {o: BasePose() for o in ORIENTACIONES}
        self.orientacion = "frente"
        self._candidata, self._t_candidata = "frente", 0.0
        self._filtros: dict[str, OneEuro] = {}
        self.parpadeos = 0
        self.bostezos = 0
        self._tiempos_parpadeo: deque[float] = deque()
        self._perclos: deque[tuple[float, bool]] = deque()
        self._reiniciar_estado()
        self.presente = False
        self._ultimo_visto = 0.0

    def _reiniciar_estado(self):
        self.estado_ojos = "abiertos"
        self._t_cierre = None
        self._alerta_cierre = False
        self._t_entrecerrado = None
        self._t_boca = None
        self.bostezando = False
        for f in self._filtros.values():
            f.reiniciar()

    # ---------- utilidades ----------
    def _f(self, nombre: str, valor: float, t: float, min_cutoff: float, beta: float) -> float:
        if nombre not in self._filtros:
            self._filtros[nombre] = OneEuro(min_cutoff, beta)
        return self._filtros[nombre](valor, t)

    def _clasificar(self, giro: float, inclinacion: float) -> str:
        if abs(giro) > C.GIRO_LIMITE:
            return "perfil_extremo"
        if abs(giro) >= C.UMBRAL_GIRO:
            return "izquierda" if giro < 0 else "derecha"
        if abs(inclinacion) >= C.UMBRAL_INCLINACION:
            return "arriba" if inclinacion < 0 else "abajo"
        return "frente"

    def _actualizar_orientacion(self, cand: str, t: float):
        """Solo cambia de orientación si la nueva se mantiene un tiempo (evita saltos)."""
        if cand == self.orientacion:
            self._candidata = cand
            return
        if cand != self._candidata:
            self._candidata, self._t_candidata = cand, t
        elif t - self._t_candidata >= C.ORIENTACION_ESTABLE_SEG:
            self.orientacion = cand

    # ---------- API ----------
    def sin_rostro(self, t: float) -> list[dict]:
        if self.presente and t - self._ultimo_visto > C.ROSTRO_PERDIDO_SEG:
            self.presente = False
            self._reiniciar_estado()
            return [{"tipo": "rostro_perdido"}]
        return []

    def procesar(self, p: np.ndarray, t: float) -> dict:
        eventos = []
        if not self.presente:
            self.presente = True
            eventos.append({"tipo": "rostro_detectado"})
        self._ultimo_visto = t

        # 1) Orientación de la cabeza (filtrada)
        giro, incl, ladeo = g.orientacion(p)
        giro = self._f("giro", giro, t, 0.8, 0.05)
        incl = self._f("incl", incl, t, 0.8, 0.05)
        ladeo = self._f("ladeo", ladeo, t, 0.8, 0.05)
        self._actualizar_orientacion(self._clasificar(giro, incl), t)
        orient = self.orientacion

        # 2) Medidas crudas -> filtradas
        ear_a = self._f("ear_a", g.ear(p, g.OJO_A), t, 1.5, 1.0)
        ear_b = self._f("ear_b", g.ear(p, g.OJO_B), t, 1.5, 1.0)
        mar = self._f("mar", g.mar(p), t, 1.0, 0.5)
        radios = {k: self._f("r_" + k, v, t, 1.0, 0.3) for k, v in g.radios(p).items()}

        # De frente se promedian ambos ojos; de lado, solo el más cercano a la cámara
        if orient in ("izquierda", "derecha"):
            cerca_a = g.profundidad_ojo(p, g.OJO_A) < g.profundidad_ojo(p, g.OJO_B)
            ear_usado, ojo_usado = (ear_a, "a") if cerca_a else (ear_b, "b")
        else:
            ear_usado, ojo_usado = (ear_a + ear_b) / 2, "ambos"

        resultado = {
            "orientacion": orient,
            "angulos": {"giro": round(giro, 1), "inclinacion": round(incl, 1), "ladeo": round(ladeo, 1)},
            "ojos": {"ear": round(ear_usado, 3), "ojo_usado": ojo_usado},
            "boca": {"mar": round(mar, 3)},
        }

        if orient == "perfil_extremo":
            resultado.update(calibrado=False, calibracion_progreso=0.0)
            resultado["ojos"]["estado"] = "no_fiable"
            resultado["boca"]["bostezando"] = False
            return self._cerrar(resultado, eventos, t)

        base = self.bases[orient]
        if not base.calibrada:
            base.agregar(ear_usado, mar, radios)
            if base.calibrada:
                eventos.append({"tipo": "calibrado", "orientacion": orient})
            resultado.update(calibrado=False, calibracion_progreso=round(base.progreso, 2))
            resultado["ojos"]["estado"] = "calibrando"
            resultado["boca"]["bostezando"] = False
            return self._cerrar(resultado, eventos, t)

        # 3) Comparación con la base de ESTA persona en ESTA orientación
        apertura = ear_usado / base.ear
        boca_extra = mar - base.mar
        radial = {k: round(v / base.radios[k], 3) for k, v in radios.items()}

        eventos += self._bostezo(boca_extra, t)
        # Al bostezar los ojos se cierran solos: no contarlo como parpadeo/somnolencia
        if not self.bostezando:
            eventos += self._ojos(apertura, t)

        if (self.estado_ojos == "abiertos" and apertura > 0.85
                and boca_extra < 0.1 and not self.bostezando):
            base.adaptar(ear_usado, mar, radios)

        resultado.update(calibrado=True, calibracion_progreso=1.0, radial=radial)
        resultado["ojos"].update(estado=self.estado_ojos, apertura_rel=round(apertura, 3),
                                 ear_base=round(base.ear, 3))
        resultado["boca"].update(mar_base=round(base.mar, 3), bostezando=self.bostezando)
        return self._cerrar(resultado, eventos, t)

    def _ojos(self, apertura: float, t: float) -> list[dict]:
        ev = []
        if self.estado_ojos == "cerrados":
            dur = t - self._t_cierre
            if apertura > C.UMBRAL_ABIERTO:
                self.estado_ojos = "abiertos"
                if dur <= C.PARPADEO_MAX_SEG:
                    self.parpadeos += 1
                    self._tiempos_parpadeo.append(t)
                    ev.append({"tipo": "parpadeo", "duracion": round(dur, 3)})
                else:
                    ev.append({"tipo": "ojos_reabiertos", "duracion": round(dur, 2)})
                self._alerta_cierre = False
            elif dur >= C.OJOS_CERRADOS_ALERTA_SEG and not self._alerta_cierre:
                self._alerta_cierre = True
                ev.append({"tipo": "ojos_cerrados_prolongado", "duracion": round(dur, 2), "capturar": True})
            return ev

        if apertura < C.UMBRAL_CERRADO:
            self.estado_ojos = "cerrados"
            self._t_cierre = t
            self._t_entrecerrado = None
            return ev

        # Entrecerrados: más pequeños de lo normal para esta persona, de forma sostenida
        if apertura < C.UMBRAL_ENTRECERRADO:
            if self._t_entrecerrado is None:
                self._t_entrecerrado = t
            elif (self.estado_ojos == "abiertos"
                  and t - self._t_entrecerrado >= C.ENTRECERRADO_MIN_SEG):
                self.estado_ojos = "entrecerrados"
                ev.append({"tipo": "ojos_entrecerrados", "apertura_rel": round(apertura, 2), "capturar": True})
        elif apertura > C.UMBRAL_ENTRECERRADO + 0.08:
            self._t_entrecerrado = None
            self.estado_ojos = "abiertos"
        return ev

    def _bostezo(self, boca_extra: float, t: float) -> list[dict]:
        ev = []
        if boca_extra > C.BOSTEZO_UMBRAL:
            if self._t_boca is None:
                self._t_boca = t
            elif not self.bostezando and t - self._t_boca >= C.BOSTEZO_MIN_SEG:
                self.bostezando = True
                self.bostezos += 1
                ev.append({"tipo": "bostezo", "capturar": True})
        elif boca_extra < C.BOSTEZO_UMBRAL * 0.6:
            if self.bostezando:
                ev.append({"tipo": "fin_bostezo", "duracion": round(t - self._t_boca, 2)})
            self._t_boca = None
            self.bostezando = False
        return ev

    def _cerrar(self, resultado: dict, eventos: list[dict], t: float) -> dict:
        ventana = C.VENTANA_ESTADISTICAS_SEG
        self._perclos.append((t, self.estado_ojos == "cerrados"))
        while self._perclos and t - self._perclos[0][0] > ventana:
            self._perclos.popleft()
        while self._tiempos_parpadeo and t - self._tiempos_parpadeo[0] > ventana:
            self._tiempos_parpadeo.popleft()
        cerrados = sum(1 for _, c in self._perclos if c)
        resultado["contadores"] = {
            "parpadeos": self.parpadeos,
            "bostezos": self.bostezos,
            "parpadeos_por_min": round(len(self._tiempos_parpadeo) * 60 / ventana, 1),
        }
        resultado["perclos"] = round(cerrados / len(self._perclos), 3)  # % de tiempo con ojos cerrados
        resultado["eventos"] = eventos
        return resultado
