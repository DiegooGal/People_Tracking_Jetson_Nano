# Sistema de Conteo de Personas con Jetson Nano

Sistema inteligente de conteo y análisis demográfico de personas en tiempo real utilizando **NVIDIA Jetson Nano**, **TensorRT** y **Deep Learning**. Diseñado para aplicaciones de analítica de audiencia en marquesinas inteligentes, espejos publicitarios y espacios comerciales.

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Nano-76B900.svg)](https://developer.nvidia.com/embedded/jetson-nano)

<img width="1497" height="762" alt="funcionamiento" src="https://github.com/user-attachments/assets/40337f9e-8042-4a99-a969-baa3b3002ce8" />

---

## Tabla de Contenidos

- [Características](#-características)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso](#-uso)
  - [Versión con GUI (Panel de Administración)](#versión-con-gui-panel-de-administración)
  - [Versión sin GUI (Línea de comandos)](#versión-sin-gui-línea-de-comandos)
- [Configuración](#-configuración)
- [Pipeline de Procesamiento](#-pipeline-de-procesamiento)
- [Modelos de Deep Learning](#-modelos-de-deep-learning)
- [Estadísticas y Métricas](#-estadísticas-y-métricas)
- [Rendimiento](#-rendimiento)
- [Solución de Problemas](#-solución-de-problemas)
- [Autores](#-autores)

---

## Características

### Funcionalidades Principales

- **Detección de Personas en Tiempo Real**: Utiliza SSD MobileNet V1 optimizado con TensorRT
- **Tracking Multi-Objeto**: Algoritmo SORT (Simple Online Realtime Tracking) con Filtros de Kalman
- **Clasificación Demográfica**: 
  - Detección de género (Male/Female)
  - Estimación de edad en 8 rangos: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- **Conteo Bidireccional**: Cuenta personas que entran y salen
- **Línea de Conteo Configurable**: Vertical u horizontal, posición ajustable
- **Análisis por Horas**: Identifica horas pico de tráfico
- **Interfaz Gráfica Moderna**: Panel de administración con diseño minimalista (tema oscuro/azul)
- **Persistencia de Datos**: Exportación a JSON con estadísticas completas
- **Optimizado para Edge**: Rendimiento de 10-15 FPS en Jetson Nano

### Dos Versiones Disponibles

1. **Versión GUI** (`main.py`): Panel de administración completo con visualización en tiempo real
2. **Versión CLI** (`usbcam_tracking_enhanced.py`): Sin interfaz gráfica, con mayor tasa de FPS

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                         JETSON NANO                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐     ┌──────────────┐    ┌─────────────┐       │
│  │ Camera   │───▶ │Preprocessing │───▶│  TensorRT   │       │
│  │  USB/CSI │     │   (Resize)   │    │  SSD-MNet   │       │
│  └──────────┘     └──────────────┘    └──────┬──────┘       │
│                                              │              │
│                                              ▼              │
│                                      ┌───────────────┐      │
│                                      │  SORT Tracker │      │
│                                      │    (Kalman)   │      │
│                                      └───────┬───────┘      │
│                                              │              │
│                         ┌────────────────────┼─────────┐    │
│                         ▼                    ▼         │    │
│                    ┌─────────────┐    ┌──────────────┐ │    │
│                    │   Age/Gender│    │ Line Crossing│ │    │
│                    │  Classifier │    │   Detection  │ │    │
│                    └─────────────┘    └──────────────┘ │    │
│                           │                    │       │    │
│                           └────────┬───────────┘       │    │
│                                    ▼                   │    │
│                            ┌───────────────┐           │    │
│                            │  Statistics   │           │    │
│                            │    Logger     │           │    │
│                            └───────────────┘           │    │
│                                    │                   │    │
│                                    ▼                   │    │
│                            ┌───────────────┐           │    │
│                            │  JSON Export  │───────────│    │
│                            └───────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Componentes Acelerados por GPU

| Componente | Hardware | Framework |
|------------|----------|-----------|
| Detección de Personas | GPU (CUDA) | TensorRT |
| Preprocesamiento | GPU (CUDA) | OpenCV + GStreamer |
| Tracking (SORT) | CPU | NumPy + FilterPy |
| Clasificación Edad/Género | CPU | OpenCV DNN |
| Interfaz Gráfica | CPU | Tkinter |

---

## Requisitos

### Hardware

- **NVIDIA Jetson Nano**
- Cámara USB o CSI (compatible con V4L2)
- Tarjeta microSD
- Fuente de alimentación 5V/4A

### Software

- **JetPack 4.6+** (Ubuntu 18.04 LTS con drivers CUDA/TensorRT)
- Python 3.6+
- OpenCV 4.1.1+ (con soporte GStreamer y CUDA)

### Dependencias Python

```
opencv-python>=4.1.1
numpy>=1.19.0
pycuda>=2020.1
Pillow>=8.0.0
filterpy>=1.4.5
scikit-learn>=0.24.0
numba>=0.53.0
```

---

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/jetson-people-counter.git
cd jetson-people-counter
```

### 2. Instalar Dependencias

```bash
# Actualizar sistema
sudo apt-get update && sudo apt-get upgrade -y

# Instalar dependencias del sistema
sudo apt-get install -y python3-pip python3-dev python3-pil python3-pil.imagetk

# Instalar dependencias de Python
pip3 install -r requirements.txt
```

### 3. Configurar TensorRT

El modelo TensorRT debe estar pre-compilado:

```bash
cd ssd/
python3 build_engine.py ssd_mobilenet_v1_coco
```

**NOTA**: Los archivos del engine TensorRT (`*.bin`) son muy grandes (>75MB) y **no están incluidos** en el repositorio por limitaciones de GitHub. Debes generarlos localmente o descargarlos desde el repositorio de Nvidia.
### 4. Descargar Modelos de Clasificación

```bash
# Modelos de edad y género
wget https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/5b60d9f8a8608cdbbcdaaa39bf28f351e8d8553b/model/age_net.caffemodel
wget https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt
wget https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_net.caffemodel
wget https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_deploy.prototxt
```

### 5. Verificar Instalación

```bash
# Ejecutar script de setup
bash setup.sh

# Verificar modelos
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
```

---

## Estructura del Proyecto

```
jetson-people-counter/
│
├── README.md                        # Este archivo
├── LICENSE                          # Licencia MIT
├── requirements.txt                 # Dependencias Python
├── setup.sh                         # Script de instalación
│
├── main.py                          # Punto de entrada (versión GUI)
├── usbcam_tracking_enhanced.py      # Versión sin GUI
├── config.py                        # Configuración global
│
├── core/                            # Lógica principal
│   ├── __init__.py
│   ├── tracker.py                   # Kalman Filter + SORT
│   ├── detector.py                  # Thread de detección TensorRT
│   ├── classifier.py                # Clasificación edad/género
│   └── fps_counter.py               # Medidor de rendimiento
│
├── gui/                             # Interfaz gráfica
│   ├── __init__.py
│   ├── main_window.py               # Ventana principal
│   ├── control_panel.py             # Panel de control
│   ├── stats_panel.py               # Panel de estadísticas
│   └── styles.py                    # Tema visual
│
├── utils/                           # Utilidades
│   ├── __init__.py
│   ├── ssd.py                       # Wrapper TensorRT SSD
│   ├── stats_logger.py              # Logger de estadísticas
│   └── geometry.py                  # Funciones geométricas (IoU, bbox)
│
├── ssd/                             # Modelos TensorRT
│   ├── build_engine.py              # Script para compilar modelos
│   ├── ssd_mobilenet_v1_coco.pb     # Modelo base
│   ├── ssd_mobilenet_v1_coco.uff    # Modelo UFF
│   ├── TRT_ssd_mobilenet_v1_coco.bin  # !!! NO INCLUIDO (>75MB)
│   └── libflattenconcat.so.*        # Plugins TensorRT
│
├── models/                          # Modelos de clasificación
│   ├── age_net.caffemodel           # Modelo de edad
│   ├── age_deploy.prototxt
│   ├── gender_net.caffemodel        # Modelo de género
│   └── gender_deploy.prototxt
│
└── people_stats.json                # Estadísticas generadas (auto)
```

---

## Uso

### Versión con GUI (Panel de Administración)

**Ejecutar:**

```bash
python3 main.py
```

**Características de la GUI:**

#### Tab 1: Control y Cámara

- **Panel Izquierdo:**
  - Checkboxes de configuración
    - Guardar Estadísticas
    - Seguimiento por Horas
    - Solo Contar Entradas
    - Clasificación Edad/Sexo
  - Sliders
    - Posición de Línea (Vertical): 0.0 (izquierda) - 1.0 (derecha)
    - Confianza de Detección: 0.3 - 0.9
  - Botones
    - ▶ Iniciar Sistema
    - ■ Detener Sistema
    - ↻ Reiniciar Contadores
  - Estadísticas en Vivo
    - FPS actual
    - Entradas / Salidas
    - Total de personas en escena

- **Panel Derecho:**
  - Vista de cámara en tiempo real
  - Visualización de bounding boxes
  - IDs de trackers
  - Información demográfica (género, edad)

#### Tab 2: Estadísticas

- **Métricas Generales:**
  - Total Entradas
  - Total Salidas
  - Neto (diferencia)
  - Total Cruces (suma)

- **Demografía:** (solo personas que entraron)
  - Distribución por género con barras visuales
  - Rangos de edad con porcentajes

- **Tráfico por Hora:**
  - Histórico de entradas/salidas por hora
  - Identificación de hora pico

**Interfaz:**

### Panel de control
<img width="1324" height="913" alt="GUI" src="https://github.com/user-attachments/assets/ed0ace04-2e74-45a3-9f9f-9d0219bc8060" />

### Estadísticas
<img width="1325" height="918" alt="Estadisticas1" src="https://github.com/user-attachments/assets/e785b6ad-8324-4509-b3d9-fff8273463bc" />
<img width="1323" height="916" alt="Estadisticas" src="https://github.com/user-attachments/assets/462a4734-e2d1-4515-94fc-4a0d6389a36d" />


---

### Versión sin GUI (Línea de Comandos)

```bash
python3 usbcam_tracking_enhanced.py
```

**Características:**
- Salida en consola con logs detallados
- Guarda video en `output.avi`
- Genera `people_stats.json` automáticamente
- Presiona `Ctrl+C` para detener

**Salida de ejemplo:**

```
Frame 120 - FPS: 12.3 - Total: 3, IN: 15
id: 42 - IN - Male - (25-32)
Frame 180 - FPS: 13.1 - Total: 2, IN: 16
Hora pico: 2026-01-17 14:00 con 23 personas
```

---

## Configuración

Edita `config.py` para personalizar el comportamiento:

### Detección

```python
DEFAULT_CONFIDENCE = 0.5      # Umbral de confianza (0.3-0.9)
INPUT_HW = (300, 300)         # Resolución de entrada
```

### Tracking

```python
MAX_TRACKERS = 30             # Máximo de personas simultáneas
MAX_AGE = 15                  # Frames antes de eliminar tracker
IOU_THRESHOLD = 0.3           # Umbral de asociación
SKIP_FRAMES = 2               # Saltar N frames (mejora FPS)
```

### Clasificación

```python
ENABLE_AGE_GENDER = True      # Activar/desactivar clasificación
AGE_GENDER_SAMPLE_RATE = 10   # Clasificar cada N frames
```

### Conteo

```python
DEFAULT_LINE_POSITION = 0.5   # Posición línea (0.0-1.0)
COUNT_ONLY_ENTERING = True    # Solo contar entradas
LINE_ORIENTATION = 'vertical' # 'vertical' o 'horizontal'
```

### Interfaz

```python
THEME = {
    'bg_dark': '#1e1e1e',
    'accent_blue': '#4a90e2',
    # ... más colores
}
```

---

## Pipeline de Procesamiento

### Flujo Completo

1. **Captura** (OpenCV VideoCapture)
   - Lee frames de cámara USB/CSI
   - Buffer mínimo para baja latencia

2. **Preprocesamiento** (CPU/GPU)
   - Resize a 300x300
   - Normalización de píxeles
   - Conversión BGR→RGB

3. **Detección** (GPU - TensorRT)
   - SSD MobileNet V1 COCO
   - Detección de clase "person" (ID=1)
   - Output: bounding boxes + confianzas

4. **Tracking** (CPU - SORT)
   - Asociación de detecciones (Hungarian Algorithm)
   - Predicción de posición (Filtro de Kalman)
   - Asignación de IDs únicos

5. **Clasificación** (CPU - OpenCV DNN) [Opcional]
   - Extracción de región de interés (ROI)
   - Clasificación de género (Male/Female)
   - Estimación de edad (8 rangos)

6. **Detección de Cruce** (CPU)
   - Comparación posición actual vs anterior
   - Detección de cruce de línea vertical/horizontal
   - Incremento de contadores

7. **Agregación** (CPU)
   - Actualización de estadísticas por hora
   - Cálculo de demografía
   - Identificación de hora pico

8. **Visualización/Exportación**
   - Renderizado en pantalla (GUI)
   - Guardado de video (CLI)
   - Exportación JSON

### Diagrama de Tiempo

```
Frame N:
├─ 0-20ms:  Captura + Preprocesamiento
├─ 20-50ms: Inferencia TensorRT (GPU)
├─ 50-60ms: Tracking SORT
├─ 60-70ms: Clasificación (async, si aplica)
├─ 70-75ms: Conteo + Estadísticas
└─ 75-80ms: Visualización
─────────────────────────────────────
Total: ~80ms → 12-13 FPS
```

---

## Modelos de Deep Learning

### 1. SSD MobileNet V1 COCO (Detección)

- **Framework**: TensorFlow → TensorRT
- **Input**: 300x300x3 (RGB)
- **Output**: 
  - Bounding boxes: [y1, x1, y2, x2] normalizado
  - Scores: confianza por detección
  - Classes: ID de clase COCO (1 = person)
- **Precisión**: mAP ~21% en COCO
- **Velocidad**: ~30ms/frame en Jetson Nano

**Conversión a TensorRT:**

```bash
cd ssd/
python3 build_engine.py ssd_mobilenet_v1_coco
```

### 2. Age Net (Estimación de Edad)

- **Arquitectura**: CNN (CaffeNet modificada)
- **Input**: 227x227x3
- **Output**: 8 clases (rangos de edad)
- **Peso**: 44MB
- **Fuente**: [GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning)

### 3. Gender Net (Clasificación de Género)

- **Arquitectura**: CNN (CaffeNet modificada)
- **Input**: 227x227x3
- **Output**: 2 clases (Male/Female)
- **Peso**: 44MB
- **Precisión**: ~60-65% en condiciones ideales

**Limitaciones Conocidas:**
- Requiere rostro frontal (±15°)
- Sensible a iluminación
- Precisión baja con oclusiones
- Entrenados con dataset de 2015 (sesgo demográfico)

---

## Estadísticas y Métricas

### Archivo JSON (`people_stats.json`)

```json
{
  "total_in": 156,
  "total_out": 42,
  "hourly_traffic": {
    "2026-01-17 09:00": {
      "in": 23,
      "out": 5
    },
    "2026-01-17 10:00": {
      "in": 45,
      "out": 12
    }
  },
  "demographics": {
    "gender": {
      "Male": 89,
      "Female": 67
    },
    "age_ranges": {
      "(15-20)": 34,
      "(25-32)": 56,
      "(38-43)": 32,
      "(48-53)": 18,
      "(60-100)": 16
    }
  },
  "start_time": "2026-01-17T08:30:00",
  "events": [
    {
      "id": 1,
      "direction": "IN",
      "gender": "Male",
      "age": "(25-32)",
      "timestamp": "2026-01-17T08:32:15"
    }
  ]
}
```

### Métricas Calculadas

| Métrica | Descripción | Uso |
|---------|-------------|-----|
| **Total Entradas** | Personas que cruzaron de izq→der | Tráfico entrante |
| **Total Salidas** | Personas que cruzaron de der→izq | Tráfico saliente |
| **Neto** | Diferencia (in - out) | Ocupación actual estimada |
| **Total Cruces** | Suma (in + out) | Tráfico total |
| **Hora Pico** | Hora con más tráfico | Planificación de recursos |
| **Demografía** | Distribución por género/edad | Targeting publicitario |

---

## Rendimiento

### Benchmarks en Jetson Nano (4GB)

| Configuración | FPS | Uso GPU | Uso RAM | Potencia |
|---------------|-----|---------|---------|----------|
| Solo Detección | 18-20 | 85% | 1.2GB | 8W |
| Detección + Tracking | 15-18 | 80% | 1.5GB | 7.5W |
| Detección + Tracking + Clasificación | 10-13 | 75% | 2.1GB | 7W |
| Con GUI | 12-15 | 70% | 2.5GB | 7.5W |

### Optimizaciones Implementadas

1. **Skip Frames**: Procesa 1 de cada N frames
2. **Thread Asíncrono**: Clasificación en paralelo
3. **Buffer Mínimo**: Cámara sin buffering
4. **TensorRT**: Inferencia optimizada en GPU
5. **Numba JIT**: Compilación de funciones críticas (IoU)
6. **Lazy Classification**: Solo clasifica trackers estables

---


## Solución de Problemas

### Error: "No se pudo abrir la cámara"

```bash
# Verificar dispositivos de video
ls -l /dev/video*

# Probar con otro índice
# En config.py:
CAMERA_INDEX = 1  # en vez de 0
```

### Error: "Engine file not found"

```bash
# Generar engine TensorRT
cd ssd/
python3 build_engine.py ssd_mobilenet_v1_coco
```

### Error: "Cannot open display"

Si usas SSH sin X11:

```bash
# Opción 1: Usar versión sin GUI
python3 usbcam_tracking_enhanced.py

# Opción 2: Habilitar X11 forwarding
ssh -X user@jetson-ip
```

### FPS Muy Bajo (<5)

```python
# En config.py, ajusta:
SKIP_FRAMES = 2 /            # Procesar 1 de cada 3 frames
ENABLE_AGE_GENDER = False    # Desactivar clasificación
MAX_TRACKERS = 20            # Reducir trackers simultáneos
```

### Detección de Género/Edad Incorrecta

**Causas comunes:**
- Ángulo no frontal (debe estar de frente ±15°)
- Iluminación pobre
- Distancia excesiva
- Rostro pequeño en imagen

**Soluciones:**
- Mejor iluminación
- Cámara más cercana
- Ajustar umbrales de confianza en `config.py`

### Error: "CUDA out of memory"

```python
# Reducir carga de GPU
MAX_TRACKERS = 15
SKIP_FRAMES = 3
```

---

## Autor

**Diego García Álvarez** - [@DiegooGal](https://github.com/DiegooGal)


**Asignatura**: Computadores Avanzados  
**Curso**: 2025-2026  
**Universidad**: Escuela Superior de Informática - UCLM  
**Profesora**: María José Santofimia Romero
