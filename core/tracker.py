#=================================================#
# tracker.py - v7                                 #
# Implementación de Kalman Filter Tracker (SORT)  #
# 19/01/2026                                      #
#=================================================#
import numpy as np
from filterpy.kalman import KalmanFilter
from utils.geometry import convert_bbox_to_z, convert_x_to_bbox


class KalmanBoxTracker:
    #=====================================================#
    # Tracker individual basado en Filtro de Kalman       #
    # Representa el estado interno de un objeto detectado #
    #=====================================================#
    
    count = 0  # Contador global de IDs
    
    def __init__(self, bbox):
        # Filtro de Kalman con modelo de velocidad constante
        # Estado: [u, v, s, r, u', v', s'] (7 dimensiones)
        # Medición: [u, v, s, r] (4 dimensiones)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # Matriz de transición de estado (modelo de movimiento)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # u = u + u'
            [0, 1, 0, 0, 0, 1, 0],  # v = v + v'
            [0, 0, 1, 0, 0, 0, 1],  # s = s + s'
            [0, 0, 0, 1, 0, 0, 0],  # r = r
            [0, 0, 0, 0, 1, 0, 0],  # u' = u'
            [0, 0, 0, 0, 0, 1, 0],  # v' = v'
            [0, 0, 0, 0, 0, 0, 1]   # s' = s'
        ])
        
        # Matriz de observación (mapeamos medición a estado)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Covarianza de ruido de medición
        self.kf.R[2:, 2:] *= 10.
        
        # Covarianza de estado inicial
        self.kf.P[4:, 4:] *= 1000.  # Alta incertidumbre en velocidades
        self.kf.P *= 10.
        
        # Covarianza de ruido de proceso
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Inicializar estado con bbox
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        # Atributos de tracking
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Atributos de demografía
        self.gender = None
        self.age_range = None
        self.last_classified_frame = -999
    
    def update(self, bbox):
        # Actualiza el estado con nueva medición
        
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self):
        # Avanza el estado y retorna la predicción
        
        # Prevenir áreas negativas
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        # Retorna el estado actual como bounding box
        
        return convert_x_to_bbox(self.kf.x)
    
    @classmethod
    def reset_count(cls):
        cls.count = 0
