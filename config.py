#=================================================#
# config.py - v3                                  #
# Configuración global del sistema People Counter #
# 20/01/2026                                      #
#=================================================#

#============================#
# CONFIGURACIÓN DE DETECCIÓN #
#============================#
INPUT_HW = (300, 300)
MODEL_NAME = 'ssd_mobilenet_v1_coco'
DEFAULT_CONFIDENCE = 0.5
MAIN_THREAD_TIMEOUT = 20.0

#===========================#
# CONFIGURACIÓN DE TRACKING #
#===========================#
MAX_TRACKERS = 30
MAX_AGE = 15
IOU_THRESHOLD = 0.3
SKIP_FRAMES = 1

#================================#
# CONFIGURACIÓN DE CLASIFICACIÓN #
#================================#
ENABLE_AGE_GENDER = True
AGE_GENDER_SAMPLE_RATE = 10

# Modelos de edad y sexo (OpenCV DNN pre-entrenados)
# Basados en: https://www.atlantic.net/gpu-server-hosting/gender-and-age-detection-using-machine-learning-on-ubuntu-24-04-server/
AGE_MODEL = 'models/age_net.caffemodel'
AGE_PROTO = 'models/age_deploy.prototxt'
GENDER_MODEL = 'models/gender_net.caffemodel'
GENDER_PROTO = 'models/gender_deploy.prototxt'

# Listas de categorías (según el modelo de OpenCV)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Parámetros de preprocesamiento
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#=========================#
# CONFIGURACIÓN DE CONTEO #
#=========================#
DEFAULT_LINE_POSITION = 0.5
COUNT_ONLY_ENTERING = True
SAVE_STATS = True
TRACK_HOURLY = True
LINE_ORIENTATION = 'vertical'

#===========================#
# CONFIGURACIÓN DE INTERFAZ #
#===========================#
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

THEME = {
    'bg_dark': '#1e1e1e',
    'bg_medium': '#2d2d2d',
    'bg_light': '#3d3d3d',
    'accent_blue': '#4a90e2',
    'accent_blue_hover': '#3a7bc8',
    'accent_red': '#e74c3c',
    'accent_gray': '#95a5a6',
    'text_color': '#e0e0e0',
    'text_dim': '#a0a0a0',
    'success': '#2ecc71',
    'warning': '#f39c12',
}

#===============================#
# CONFIGURACIÓN DE ESTADÍSTICAS #
#===============================#
STATS_FILENAME = 'people_stats.json'
MAX_EVENTS_STORED = 1000
STATS_SAVE_INTERVAL = 600  # frames

#=========================#
# CONFIGURACIÓN DE CÁMARA #
#=========================#
CAMERA_INDEX = 0
CAMERA_BUFFER_SIZE = 1
CAMERA_FPS = 30