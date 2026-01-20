#=================================================#
# fps_counter.py - v1                             #
# Contador de FPS en tiempo real                  #
# 18/01/2026                                      #
#=================================================#
import time
import collections


class FPSCounter:
    # Calcula FPS promedio usando ventana deslizante
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = collections.deque(maxlen=window_size)
        self.last_time = time.time()
        
    def update(self):
        # Registra un nuevo frame
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
    def get_fps(self):
        # Calcula y retorna FPS promedio
        if len(self.frame_times) == 0:
            return 0.0
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        if avg_frame_time == 0:
            return 0.0
            
        return 1.0 / avg_frame_time
    
    def reset(self):
        self.frame_times.clear()
        self.last_time = time.time()
