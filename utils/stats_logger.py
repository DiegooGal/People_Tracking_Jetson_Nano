#=================================================#
# stats_logger.py - v3                            #
# Logger de estadísticas del sistema              #
# 18/01/2026                                      #
#=================================================#
import json
from datetime import datetime
import config


class StatisticsLogger:
    # Gestiona estadísticas y persistencia en JSON
    
    def __init__(self, filename=None):
        self.filename = filename or config.STATS_FILENAME
        self.stats = self._init_stats()
        
    def _init_stats(self):
        return {
            'total_in': 0,
            'total_out': 0,
            'hourly_traffic': {},
            'demographics': {
                'gender': {},
                'age_ranges': {}
            },
            'start_time': datetime.now().isoformat(),
            'events': []
        }
    
    def reset(self):
        self.stats = self._init_stats()
        
    def log_entry(self, person_id, direction, timestamp=None, gender=None, age_range=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')
        
        # Actualizar totales
        if direction == 'IN':
            self.stats['total_in'] += 1
            
            # Demografía solo para entradas y solo si tiene valor válido
            if gender and gender not in ['None', 'Desconocido', None]:
                if gender not in self.stats['demographics']['gender']:
                    self.stats['demographics']['gender'][gender] = 0
                self.stats['demographics']['gender'][gender] += 1
                
            if age_range and age_range not in ['None', 'Desconocido', None]:
                if age_range not in self.stats['demographics']['age_ranges']:
                    self.stats['demographics']['age_ranges'][age_range] = 0
                self.stats['demographics']['age_ranges'][age_range] += 1
                
        elif direction == 'OUT':
            self.stats['total_out'] += 1
        
        # Tráfico por hora
        if hour_key not in self.stats['hourly_traffic']:
            self.stats['hourly_traffic'][hour_key] = {'in': 0, 'out': 0}
        
        if direction == 'IN':
            self.stats['hourly_traffic'][hour_key]['in'] += 1
        elif direction == 'OUT':
            self.stats['hourly_traffic'][hour_key]['out'] += 1
        
        # Eventos (limitado para no saturar memoria)
        if len(self.stats['events']) < config.MAX_EVENTS_STORED:
            event = {
                'id': person_id,
                'direction': direction,
                'timestamp': timestamp.isoformat()
            }
            if gender:
                event['gender'] = gender
            if age_range:
                event['age'] = age_range
            self.stats['events'].append(event)
    
    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.stats, f, indent=2)
            return True
        except Exception as e:
            print(f"Error guardando estadísticas: {e}")
            return False
    
    def load(self):
        try:
            with open(self.filename, 'r') as f:
                self.stats = json.load(f)
            return True
        except FileNotFoundError:
            print(f"Archivo {self.filename} no encontrado")
            return False
        except Exception as e:
            print(f"Error cargando estadísticas: {e}")
            return False
    
    def get_peak_hour(self):
        if not self.stats['hourly_traffic']:
            return None, 0
        
        max_traffic = 0
        peak_hour = None
        
        for hour, data in self.stats['hourly_traffic'].items():
            total = data['in'] + data['out']
            if total > max_traffic:
                max_traffic = total
                peak_hour = hour
        
        return peak_hour, max_traffic
    
    def get_summary(self):
        peak_hour, peak_traffic = self.get_peak_hour()
        
        return {
            'total_in': self.stats['total_in'],
            'total_out': self.stats['total_out'],
            'net': self.stats['total_in'] - self.stats['total_out'],
            'peak_hour': peak_hour,
            'peak_traffic': peak_traffic,
            'demographics': self.stats['demographics'],
            'hourly_traffic': self.stats['hourly_traffic']
        }
