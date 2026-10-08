# Identificación con webcam USB (Python)

Captura vídeo de la webcam, detecta lo que haya (por ahora **rostros** con OpenCV) y
deja los resultados en `salida/` para que **una IA local los lea más adelante**.

```
Webcam USB ──> src/camara.py ──> Identificador ──> src/salida.py ──> salida/detecciones.jsonl
                                  │                                  salida/capturas/*.jpg
                                  ├─ rostros_opencv.py  (ahora)              │
                                  └─ ia_local.py        (futuro)             ▼
                                                                  IA local / leer_detecciones.py
```

## Instalación (Windows)

```powershell
cd webcam_identificacion
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Se fija OpenCV 4.x: en OpenCV 5 el detector Haar (`CascadeClassifier`) ya no viene en el paquete principal.

## Uso

```powershell
python main.py --listar          # ver qué índices de cámara funcionan
python main.py                   # webcam USB (índice 1), con ventana  (q = salir, c = guardar captura)
python main.py --camara 0        # cámara integrada del portátil
python main.py --sin-ventana     # solo registrar, sin mostrar vídeo
python leer_detecciones.py       # (otra terminal) ver los eventos en vivo, como lo hará la IA
```

Toda la configuración está en [src/config.py](src/config.py).

## Formato de salida (contrato con la IA)

`salida/detecciones.jsonl`: una línea JSON por cada frame analizado en el que se detectó algo:

```json
{"fecha": "2026-09-30T19:10:54.049", "fuente": "rostros_opencv", "ancho": 1280, "alto": 720,
 "imagen": "capturas/20260930_191054_049591.jpg",
 "detecciones": [{"etiqueta": "rostro", "confianza": 1.0, "caja": [412, 180, 210, 210], "extra": {}}]}
```

- `caja` = `[x, y, ancho, alto]` en píxeles.
- `imagen` puede ser `null` (las capturas se guardan como máximo cada `SEGUNDOS_ENTRE_CAPTURAS`).

## Integración con IA Local (Ollama + Moondream)

El proyecto incluye soporte para **Moondream** (VLM ligero), ideal para correr en la Raspberry Pi 5 o en local:

### 1. Análisis directo de imágenes o capturas
```powershell
# Analizar imagen de prueba con Moondream
python analizar_moondream.py ./prueba.jpg

# Preguntar algo específico
python analizar_moondream.py ./prueba.jpg --prompt "Describe la imagen y di si la persona tiene los ojos cerrados"

# Tomar una foto con la webcam y analizarla al instante
python analizar_moondream.py --capturar
```

### 2. Modo en vivo con la webcam y Moondream
```powershell
python main.py --modo ia
```
- Corre a 30 FPS en la cámara sin congelar la ventana.
- Consulta Moondream de forma asíncrona cada 3 segundos.
- Si detecta ojos cerrados, emite alerta visual y guarda la captura.

### 3. Consumidor de eventos en tiempo real
```powershell
# Escucha eventos y analiza cada captura guardada con Moondream
python leer_detecciones.py
```

### Configuración de Red (Windows <-> Raspberry Pi)
Si ejecutas el código en Windows y Ollama está corriendo en la Raspberry Pi, el script detecta automáticamente la IP `192.168.79.28:11434` o puedes fijar la variable de entorno:
```powershell
$env:OLLAMA_HOST="http://192.168.79.28:11434"
```

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `No se pudo abrir la cámara` / "sin imagen" | Otra app la usa (pestaña del navegador, Discord, Teams, OBS). Ciérrala y prueba `--listar`. |
| Imagen negra | Tapa de privacidad o lente tapada, o es otra cámara (portátil vs USB). |
| Windows bloquea la cámara | Configuración → Privacidad → Cámara → permitir apps de escritorio. |
