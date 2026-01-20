#=================================================#
# detector.py - v6                                #
# Thread de detección con TensorRT                #
# 20/01/2026                                      #
#=================================================#
import threading
import cv2
import pycuda.driver as cuda
from utils.ssd import TrtSSD
import config


class DetectionThread(threading.Thread):
    # Thread para ejecutar detección TensorRT en paralelo
    
    def __init__(self, condition, camera, model_name, confidence_threshold):
        threading.Thread.__init__(self)
        self.daemon = True
        
        self.condition = condition
        self.camera = camera
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        self.cuda_ctx = None
        self.trt_ssd = None
        self.running = False
        
        # Variables compartidas (se actualizan en cada frame)
        self.current_image = None
        self.current_boxes = None
    
    def run(self):
        print(f'DetectionThread: Cargando modelo {self.model_name}...')
        
        # Inicializar contexto CUDA
        self.cuda_ctx = cuda.Device(0).make_context()
        
        # Cargar modelo TensorRT
        self.trt_ssd = TrtSSD(self.model_name, config.INPUT_HW)
        
        print('DetectionThread: Modelo cargado, iniciando detección...')
        self.running = True
        
        frame_skip_counter = 0
        
        while self.running:
            # Capturar frame
            ret, img = self.camera.read()
            
            if img is None:
                continue
            
            # Skip frames si está configurado
            frame_skip_counter += 1
            if frame_skip_counter < config.SKIP_FRAMES:
                continue
            frame_skip_counter = 0
            
            # Redimensionar para el modelo
            img = cv2.resize(img, config.INPUT_HW, interpolation=cv2.INTER_LINEAR)
            
            # Ejecutar detección
            boxes, confs, clss = self.trt_ssd.detect(img, self.confidence_threshold)
            
            # Actualizar variables compartidas de forma segura
            with self.condition:
                self.current_image = img
                self.current_boxes = boxes
                self.condition.notify()
        
        # Limpieza
        del self.trt_ssd
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('DetectionThread: Detenido')
    
    def stop(self):
        self.running = False
        self.join(timeout=2.0)
    
    def get_latest_detection(self):
        with self.condition:
            return self.current_image, self.current_boxes
    
    def update_confidence(self, new_confidence):
        self.confidence_threshold = max(0.3, min(0.9, new_confidence))
