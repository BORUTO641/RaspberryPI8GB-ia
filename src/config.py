"""Configuración central del proyecto. Cambia aquí los valores, no en el código."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# --- Cámara ---
INDICE_CAMARA = 1          # 1 = webcam USB (0 = cámara integrada del portátil). Usa `python main.py --listar` para ver otras.
ANCHO = 1280
ALTO = 720
FPS = 30
ESPEJO = True              # voltear la imagen como un espejo (izquierda = izquierda de la persona)

# --- Identificación ---
# "gestos"  -> ojos, bostezo, orientación y punto radial con MediaPipe (recomendado)
# "rostros" -> solo recuadro de rostro con OpenCV (rápido, sin gestos)
# "ia"      -> modelo de visión local con Ollama Moondream (src/identificadores/ia_local.py)
MODO_IDENTIFICADOR = "gestos"
ANALIZAR_CADA_N_FRAMES = 1  # gestos necesita todos los frames para no perder parpadeos
RUTA_MODELO_ROSTRO = RAIZ / "modelos" / "face_landmarker.task"

# --- IA Local (Ollama + Moondream) ---
OLLAMA_MODELO = "moondream"
OLLAMA_PROMPT_DEFECTO = "Describe la imagen y di si la persona tiene los ojos cerrados"
OLLAMA_INTERVALO_SEG = 3.0     # segundos entre consultas periódicas al modelo en el loop de vídeo
OLLAMA_HOST = None             # None = detección automática (localhost o Pi remota)


# --- Gestos: calibración por persona y por orientación ---
CALIBRACION_MUESTRAS = 45      # ~2-3 s mirando normal en cada orientación
ADAPTACION_ALPHA = 0.005       # qué tan rápido se ajusta la base después (0 = nunca)

# --- Gestos: orientación de la cabeza (grados) ---
UMBRAL_GIRO = 20               # a partir de aquí se considera "izquierda"/"derecha"
UMBRAL_INCLINACION = 15        # a partir de aquí "arriba"/"abajo"
GIRO_LIMITE = 55               # más girado que esto: perfil extremo, los ojos no son fiables
ORIENTACION_ESTABLE_SEG = 0.3  # tiempo mínimo antes de aceptar un cambio de orientación

# --- Gestos: ojos (en % de la apertura normal de ESA persona) ---
UMBRAL_CERRADO = 0.55          # por debajo -> cerrados
UMBRAL_ABIERTO = 0.70          # por encima -> se reabren (histéresis: evita parpadeos falsos)
UMBRAL_ENTRECERRADO = 0.80     # por debajo de forma sostenida -> entrecerrados
ENTRECERRADO_MIN_SEG = 1.0
PARPADEO_MAX_SEG = 0.5         # cierre más corto que esto = parpadeo
OJOS_CERRADOS_ALERTA_SEG = 1.5 # cierre más largo que esto = alerta (posible somnolencia)

# --- Gestos: boca ---
BOSTEZO_UMBRAL = 0.35          # apertura de boca extra sobre la base de la persona
BOSTEZO_MIN_SEG = 1.0          # hablar abre la boca poco tiempo; bostezar, más de 1 s

# --- Gestos: otros ---
ROSTRO_PERDIDO_SEG = 0.5
VENTANA_ESTADISTICAS_SEG = 60  # parpadeos/min y PERCLOS sobre el último minuto

# --- Salida (lo que leerá la IA) ---
CARPETA_SALIDA = RAIZ / "salida"
ARCHIVO_DETECCIONES = CARPETA_SALIDA / "detecciones.jsonl"
CARPETA_CAPTURAS = CARPETA_SALIDA / "capturas"
INTERVALO_REGISTRO_SEG = 1.0   # estado periódico; los eventos se escriben siempre al instante
GUARDAR_CAPTURA_AL_DETECTAR = True
SEGUNDOS_ENTRE_CAPTURAS = 2.0  # evita llenar el disco con miles de imágenes
