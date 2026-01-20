#================================================================================================================================#
# classifier.py - v12                                                                                                            #
# Clasificador de edad y sexo.                                                                                                   #
# Basado en: https://www.atlantic.net/gpu-server-hosting/gender-and-age-detection-using-machine-learning-on-ubuntu-24-04-server/ #
# 20/01/2026                                                                                                                     #
#================================================================================================================================#

import threading
import queue
import cv2
import numpy as np
import config


class AgeGenderClassifier(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        
        self.running = False
        self.queue = queue.Queue(maxsize=10)
        
        # Redes neuronales
        self.age_net = None
        self.gender_net = None
        
        # Detector de rostros opcional (mejora precisión)
        self.face_net = None
        
    def load_models(self):
        #==========================================#
        # Carga modelos de edad y sexo desde disco #
        #==========================================#
        
        try:
            print("Cargando modelos de clasificación...")
            
            # Cargar modelo de género
            self.gender_net = cv2.dnn.readNet(
                config.GENDER_MODEL,
                config.GENDER_PROTO
            )
            
            # Cargar modelo de edad
            self.age_net = cv2.dnn.readNet(
                config.AGE_MODEL,
                config.AGE_PROTO
            )
            
            print("Modelos de edad/sexo cargados correctamente")
            return True
            
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            print(" Asegúrate de tener los archivos:")
            print(f"  - {config.AGE_MODEL}")
            print(f"  - {config.AGE_PROTO}")
            print(f"  - {config.GENDER_MODEL}")
            print(f"  - {config.GENDER_PROTO}")
            return False
    
    def classify_face(self, face_img):
        #===============================================#
        # Clasifica edad y sexo de una imagen           #
        # Implementación según tutorial de atlantic.net #
        #===============================================#
        
        if self.age_net is None or self.gender_net is None:
            return None, None
        
        try:
            # Obtener dimensiones
            height, width = face_img.shape[:2]
            
            # Verificar tamaño mínimo
            if height < 20 or width < 20:
                return None, None
            
            # Preparar blob para la red neuronal
            # Según el tutorial: 227x227, MODEL_MEAN_VALUES, swapRB=False
            blob = cv2.dnn.blobFromImage(
                face_img,
                1.0,
                (227, 227),
                config.MODEL_MEAN_VALUES,
                swapRB=False
            )
            
            # ===== PREDICCIÓN DE GÉNERO =====
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            
            # Obtener índice con mayor probabilidad
            gender_idx = gender_preds[0].argmax()
            gender = config.GENDER_LIST[gender_idx]
            gender_confidence = gender_preds[0][gender_idx]
            
            # Solo aceptar si confianza > 60%
            if gender_confidence < 0.60:
                gender = None
            
            # ===== PREDICCIÓN DE EDAD =====
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            
            # Obtener índice con mayor probabilidad
            age_idx = age_preds[0].argmax()
            age_range = config.AGE_LIST[age_idx]
            age_confidence = age_preds[0][age_idx]
            
            # Solo aceptar si confianza > 50%
            if age_confidence < 0.50:
                age_range = None
            
            return gender, age_range
            
        except Exception as e:
            print(f"Error en clasificación: {e}")
            return None, None
    
    def run(self):
        self.running = True
        
        while self.running:
            try:
                # Obtener tarea de la cola (timeout 0.1s)
                task = self.queue.get(timeout=0.1)
                
                if task is None:
                    continue
                
                tracker_id, face_img, callback = task
                
                # Clasificar
                gender, age_range = self.classify_face(face_img)
                
                # Llamar callback con resultados
                if callback:
                    callback(tracker_id, gender, age_range)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en thread de clasificación: {e}")
    
    def classify_async(self, tracker_id, face_img, callback):
        #========================================#
        # Añade tarea de clasificación a la cola #
        #========================================#
        
        try:
            self.queue.put_nowait((tracker_id, face_img, callback))
        except queue.Full:
            # Si la cola está llena, ignorar (no crítico)
            pass
    
    def stop(self):
        self.running = False
        self.join(timeout=1.0)