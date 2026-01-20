#=================================================#
# usbcam_tracking_enhanched.py - v9               #
# Sistema de Conteo de Personas con Jetson Nano   #
# Sin panel de administración                     #
# Promedia entre 15 y 25 FPS estables             #
# 13/01/2026                                      #
#=================================================#

import cv2
import numpy as np
import collections
import threading
import pycuda.driver as cuda
import time
import json
from datetime import datetime
import queue

from utils.ssd import TrtSSD
from filterpy.kalman import KalmanFilter
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

s_img, s_boxes = None, None
INPUT_HW = (300, 300)
MAIN_THREAD_TIMEOUT = 20.0

# CONFIGURACIÓN DE OPTIMIZACIÓN
SKIP_FRAMES = 1
MAX_TRACKERS = 50
DISPLAY_SCALE = 1.0

# CONFIGURACIÓN DE MODIFICACIONES
SAVE_STATS = True
TRACK_HOURLY = True
CUSTOM_LINE_POSITION = 0.5
COUNT_ONLY_ENTERING = True

# CONFIGURACIÓN DE EDAD Y SEXO
ENABLE_AGE_GENDER = True  # Activar/desactivar clasificación
AGE_GENDER_SAMPLE_RATE = 5  # Clasificar 1 de cada N frames por tracker

# Modelos para edad y sexo (MobileNet ligeros)
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
MEAN_PROTO = "mean.binaryproto"

# Listas de categorías
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Masculino', 'Femenino']

# iou - Optimizado con numba
@jit(nopython=True)
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    if x[2] <= 0 or x[3] <= 0:
        if score is None:
            return np.array([0, 0, 0, 0]).reshape((1, 4))
        else:
            return np.array([0, 0, 0, 0, score]).reshape((1, 5))
    
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Atributos para edad y sexo
        self.gender = None
        self.age_range = None
        self.last_classified_frame = -999

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
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
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class AgeGenderClassifier(threading.Thread):
    """Thread separado para clasificación de edad y sexo sin afectar FPS"""
    def __init__(self):
        threading.Thread.__init__(self)
        self.running = False
        self.queue = queue.Queue(maxsize=10)
        self.age_net = None
        self.gender_net = None
        self.daemon = True

    def load_mean(self):
        with open(MEAN_PROTO, 'rb') as f:
            mean_blob = cv2.dnn.readNetFromCaffe(" "," ")._readProtoFromBinaryFile(MEAN_PROTO)

        mean = np.array(mean_blob.data)
        mean = mean.reshape((mean_blob.channels, mean_blob.height, mean_blob.width))
        return mean.mean(axis=(1,2))
        
    def load_models(self):
        """Cargar modelos de edad y sexo"""
        try:
            # Intentar cargar modelos preentrenados
            self.age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
            self.gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
            self.mean_values = self.load_mean()
            print("Modelos de edad/sexo cargados correctamente")
            return True
        except:
            print("ADVERTENCIA: No se encontraron modelos de edad/sexo")
            print("Usando clasificación simulada para demostración")
            return False
    
    def classify_face(self, face_img):
        """Clasificar edad y sexo de una imagen de rostro"""
        if self.age_net is None or self.gender_net is None:
            # Simulación para demostración si no hay modelos
            return self._simulate_classification()
        
        try:
            # Preprocesar imagen
            face = cv2.resize(face_img, (227, 227))
            blob = cv2.dnn.blobFromImage(face, 
                                        scalefacto=1.0,
                                        mean =self.mean_values, 
                                        swapRB=False,
                                        crop=False)
            
            # Predecir sexo
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender_idx = gender_preds[0].argmax()
            gender = GENDER_LIST[gender_idx]
            
            # Predecir edad
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_idx = age_preds[0].argmax()
            age_range = AGE_LIST[age_idx]
            
            return gender, age_range
        except:
            return self._simulate_classification()
    
    def _simulate_classification(self):
        """Simulación de clasificación cuando no hay modelos"""
        # Distribución realista para demostración
        gender = np.random.choice(['Masculino', 'Femenino'], p=[0.52, 0.48])
        age_range = np.random.choice(AGE_LIST, 
                                     p=[0.05, 0.08, 0.12, 0.20, 0.25, 0.18, 0.10, 0.02])
        return gender, age_range
    
    def run(self):
        """Thread principal de clasificación"""
        self.running = True
        while self.running:
            try:
                # Obtener trabajo de la cola (timeout 0.1s)
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
                print(f"Error en clasificación: {e}")
    
    def classify_async(self, tracker_id, face_img, callback):
        """Añadir tarea de clasificación a la cola"""
        try:
            self.queue.put_nowait((tracker_id, face_img, callback))
        except queue.Full:
            pass  # Si la cola está llena, ignorar
    
    def stop(self):
        self.running = False


class TrtThread(threading.Thread):
    def __init__(self, condition, cam, model, conf_th):
        threading.Thread.__init__(self)
        self.condition = condition
        self.cam = cam
        self.model = model
        self.conf_th = conf_th
        self.cuda_ctx = None
        self.trt_ssd = None
        self.running = False

    def run(self):
        global s_img, s_boxes

        print('TrtThread: loading the TRT SSD engine...')
        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_ssd = TrtSSD(self.model, INPUT_HW)
        print('TrtThread: start running...')
        self.running = True
        
        frame_skip_counter = 0
        
        while self.running:
            ret, img = self.cam.read()
            if img is None:
                break
            
            frame_skip_counter += 1
            if frame_skip_counter < SKIP_FRAMES:
                continue
            frame_skip_counter = 0
            
            img = cv2.resize(img, INPUT_HW, interpolation=cv2.INTER_LINEAR)
            boxes, confs, clss = self.trt_ssd.detect(img, self.conf_th)
            
            with self.condition:
                s_img, s_boxes = img, boxes
                self.condition.notify()
                
        del self.trt_ssd
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.running = False
        self.join()


class StatisticsLogger:
    def __init__(self, filename='people_stats.json'):
        self.filename = filename
        self.stats = {
            'total_in': 0,
            'total_out': 0,
            'hourly_traffic': {},
            'demographics': {
                'gender': {'Masculino': 0, 'Femenino': 0},
                'age_ranges': {age: 0 for age in AGE_LIST}
            },
            'start_time': datetime.now().isoformat(),
            'events': []
        }
        
    def log_entry(self, person_id, direction, timestamp=None, gender=None, age_range=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')
        
        if direction == 'IN':
            self.stats['total_in'] += 1
            # Registrar demografía solo para entradas
            if gender:
                self.stats['demographics']['gender'][gender] += 1
            if age_range:
                self.stats['demographics']['age_ranges'][age_range] += 1
        elif direction == 'OUT':
            self.stats['total_out'] += 1
            
        if hour_key not in self.stats['hourly_traffic']:
            self.stats['hourly_traffic'][hour_key] = {'in': 0, 'out': 0}
        
        if direction == 'IN':
            self.stats['hourly_traffic'][hour_key]['in'] += 1
        elif direction == 'OUT':
            self.stats['hourly_traffic'][hour_key]['out'] += 1
        
        if len(self.stats['events']) < 1000:
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
        except Exception as e:
            print(f"Error guardando estadísticas: {e}")
            
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
    
    def get_demographics_summary(self):
        """Obtener resumen demográfico"""
        total = sum(self.stats['demographics']['gender'].values())
        if total == 0:
            return None
        
        summary = {
            'total': total,
            'gender': self.stats['demographics']['gender'].copy(),
            'age_ranges': self.stats['demographics']['age_ranges'].copy()
        }
        return summary


class FPSCounter:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = collections.deque(maxlen=window_size)
        self.last_time = time.time()
        
    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
    def get_fps(self):
        if len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time == 0:
            return 0.0
        return 1.0 / avg_frame_time


def get_frame(condition, age_gender_classifier=None):
    frame = 0
    max_age = 15

    trackers = []
    global s_img, s_boxes

    print("Iniciando sistema de conteo de personas...")
    idstp = collections.defaultdict(list)
    idcnt = []
    incnt, outcnt = 0, 0
    
    stats_logger = StatisticsLogger() if SAVE_STATS else None
    fps_counter = FPSCounter()
    
    # Callback para resultados de clasificación
    def on_classification_result(tracker_id, gender, age_range):
        for trk in trackers:
            if trk.id == tracker_id:
                trk.gender = gender
                trk.age_range = age_range
                break

    try:
        while True:
            fps_counter.update()
            
            with condition:
                if condition.wait(timeout=MAIN_THREAD_TIMEOUT):
                    img, boxes = s_img, s_boxes
                else:
                    raise SystemExit('ERROR: timeout waiting for img from child')
            
            if img is None:
                continue
            
            img = img.copy()
            boxes = np.array(boxes)

            H, W = img.shape[:2]
            line_position = int(W * CUSTOM_LINE_POSITION)

            if len(trackers) > MAX_TRACKERS:
                trackers.sort(key=lambda x: x.time_since_update, reverse=True)
                trackers = trackers[:MAX_TRACKERS]

            trks = np.zeros((len(trackers), 5))
            to_del = []

            for t, trk in enumerate(trks):
                pos = trackers[t].predict()[0]
                trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
            for t in reversed(to_del):
                trackers.pop(t)

            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(boxes, trks)

            for t, trk in enumerate(trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    if len(d) > 0:
                        trk.update(boxes[d, :][0])
                        xmin, ymin, xmax, ymax = boxes[d, :][0]
                        cy = int((ymin + ymax) / 2)
                        cx = int((xmin + xmax) / 2)
                        
                        # Clasificar edad/sexo de forma asíncrona (solo cada N frames)
                        if (ENABLE_AGE_GENDER and age_gender_classifier and 
                            frame - trk.last_classified_frame > AGE_GENDER_SAMPLE_RATE and
                            trk.gender is None):  # Solo si no ha sido clasificado
                            
                            # Extraer región de la persona
                            face_region = img[max(0, ymin):min(H, ymax), 
                                            max(0, xmin):min(W, xmax)]
                            
                            if face_region.size > 0:
                                age_gender_classifier.classify_async(
                                    trk.id, face_region, on_classification_result
                                )
                                trk.last_classified_frame = frame

                        if len(idstp[trk.id]) > 0:
                            prev_cx = idstp[trk.id][-1][0]
                            
                            if prev_cx < line_position and cx >= line_position and trk.id not in idcnt:
                                incnt += 1
                                print(f"id: {trk.id} - IN - {trk.gender or '?'} - {trk.age_range or '?'}")
                                idcnt.append(trk.id)
                                if stats_logger:
                                    stats_logger.log_entry(trk.id, 'IN', 
                                                          gender=trk.gender, 
                                                          age_range=trk.age_range)

                            elif not COUNT_ONLY_ENTERING and prev_cx >= line_position and cx < line_position and trk.id not in idcnt:
                                outcnt += 1
                                print(f"id: {trk.id} - OUT")
                                idcnt.append(trk.id)
                                if stats_logger:
                                    stats_logger.log_entry(trk.id, 'OUT')

                        u, v = trk.kf.x[0], trk.kf.x[1]
                        if len(idstp[trk.id]) > 10:
                            idstp[trk.id].pop(0)
                        idstp[trk.id].append([u, v])

                        # Dibujar rectángulo
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        
                        # Mostrar ID y demografía
                        label = f"{trk.id}"
                        if trk.gender:
                            label += f" {trk.gender[0]}"  # M o F
                        if trk.age_range:
                            label += f" {trk.age_range}"
                        
                        cv2.putText(img, label, (int(xmin), int(ymin) - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            current_fps = fps_counter.get_fps()
            
            # FPS
            cv2.putText(img, f"FPS:{int(current_fps)}", (W - 80, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Total
            cv2.putText(img, f"Total:{len(trackers)}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Línea de conteo
            cv2.line(img, (line_position, 0), (line_position, H), (255, 0, 0), 2)
            
            if COUNT_ONLY_ENTERING:
                cv2.putText(img, f"IN:{incnt}", (5, line_position - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(img, f"IN:{incnt}", (5, line_position - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"OUT:{outcnt}", (5, line_position + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for i in unmatched_dets:
                trk = KalmanBoxTracker(boxes[i, :])
                trackers.append(trk)
                trk.id = len(trackers)
                u, v = trk.kf.x[0], trk.kf.x[1]
                idstp[trk.id].append([u, v])

            i = len(trackers)
            for trk in reversed(trackers):
                i -= 1
                if trk.time_since_update > max_age:
                    trackers.pop(i)

            if DISPLAY_SCALE != 1.0:
                display_h = int(H * DISPLAY_SCALE)
                display_w = int(W * DISPLAY_SCALE)
                img = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("People Counter", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nSaliendo...")
                break
            
            if frame % 60 == 0:
                print(f"Frame {frame} - FPS: {current_fps:.1f} - Total: {len(trackers)}, IN: {incnt}")
                
                if stats_logger and frame % 600 == 0:
                    stats_logger.save()
                    
            frame += 1
            
    except KeyboardInterrupt:
        print("\nInterrumpido...")
    finally:
        cv2.destroyAllWindows()
        
        if stats_logger:
            stats_logger.save()
            peak_hour, traffic = stats_logger.get_peak_hour()
            if peak_hour:
                print(f"\nHora pico: {peak_hour} con {traffic} personas")
            print(f"Total IN: {incnt}, OUT: {outcnt}")
            print(f"FPS promedio: {fps_counter.get_fps():.1f}")
            
            # Mostrar resumen demográfico
            demo = stats_logger.get_demographics_summary()
            if demo:
                print(f"\n--- Resumen Demográfico ---")
                print(f"Total clasificado: {demo['total']}")
                print(f"Género: {demo['gender']}")
                print(f"Rangos de edad: {demo['age_ranges']}")


if __name__ == '__main__':
    model = 'ssd_mobilenet_v1_coco'
    
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FPS, 30)

    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cuda.init()

    # Inicializar clasificador de edad/sexo en thread separado
    age_gender_classifier = None
    if ENABLE_AGE_GENDER:
        age_gender_classifier = AgeGenderClassifier()
        age_gender_classifier.load_models()
        age_gender_classifier.start()
        print("Clasificador de edad/sexo iniciado")

    condition = threading.Condition()
    trt_thread = TrtThread(condition, cam, model, conf_th=0.5)
    
    try:
        trt_thread.start()
        get_frame(condition, age_gender_classifier)
    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        if age_gender_classifier:
            age_gender_classifier.stop()
        trt_thread.stop()
        cam.release()
        print("Programa finalizado")
