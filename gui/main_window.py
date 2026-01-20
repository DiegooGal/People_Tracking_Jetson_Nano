#=================================================#
# main_window.py - v3                             #
# Ventana principal de la aplicación              #
# 18/01/2026                                      #
#=================================================#
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL
from PIL import Image
import PIL.ImageTk as ImageTk
import threading
import numpy as np
import collections

import config
from gui.styles import setup_styles
from gui.control_panel import ControlPanel
from gui.stats_panel import StatsPanel
from core.tracker import KalmanBoxTracker
from core.detector import DetectionThread
from core.classifier import AgeGenderClassifier
from core.fps_counter import FPSCounter
from utils.stats_logger import StatisticsLogger
from utils.geometry import associate_detections_to_trackers
import pycuda.driver as cuda


class MainWindow:
    # Ventana principal con toda la lógica de la aplicación
    
    def __init__(self, root):
        self.root = root
        self.root.title("Panel de Administración - People Counter")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.configure(bg=config.THEME['bg_dark'])
        
        # Setup estilos
        setup_styles()
        
        # Variables de configuración (tk.Variables)
        self.save_stats = tk.BooleanVar(value=config.SAVE_STATS)
        self.track_hourly = tk.BooleanVar(value=config.TRACK_HOURLY)
        self.count_only_entering = tk.BooleanVar(value=config.COUNT_ONLY_ENTERING)
        self.enable_age_gender = tk.BooleanVar(value=config.ENABLE_AGE_GENDER)
        self.line_position = tk.DoubleVar(value=config.DEFAULT_LINE_POSITION)
        self.confidence = tk.DoubleVar(value=config.DEFAULT_CONFIDENCE)
        
        # Sistema de tracking
        self.is_running = False
        self.camera = None
        self.detection_thread = None
        self.classifier_thread = None
        self.condition = None
        
        # Estado del tracking
        self.trackers = []
        self.idstp = collections.defaultdict(list)
        self.idcnt = []
        self.incnt = 0
        self.outcnt = 0
        
        # Utilidades
        self.fps_counter = FPSCounter()
        self.stats_logger = StatisticsLogger()
        
        # Crear interfaz
        self.create_ui()
        
        # Configurar cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_ui(self):
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Control y Cámara
        tab_control = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(tab_control, text="  Control y Cámara  ")
        self.control_panel = ControlPanel(tab_control, self)
        
        # Tab 2: Estadísticas
        tab_stats = ttk.Frame(self.notebook, style="Dark.TFrame")
        self.notebook.add(tab_stats, text="  Estadísticas  ")
        self.stats_panel = StatsPanel(tab_stats, self)
    
    def start_system(self):
        if self.is_running:
            return
        
        try:
            # Inicializar cámara
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
            self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            if not self.camera.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la cámara")
                return
            
            # Inicializar CUDA
            cuda.init()
            
            # Iniciar thread de detección
            self.condition = threading.Condition()
            self.detection_thread = DetectionThread(
                self.condition,
                self.camera,
                config.MODEL_NAME,
                self.confidence.get()
            )
            self.detection_thread.start()
            
            # Iniciar classifier si está habilitado
            if self.enable_age_gender.get():
                self.classifier_thread = AgeGenderClassifier()
                self.classifier_thread.load_models()
                self.classifier_thread.start()
            
            self.is_running = True
            self.control_panel.enable_stop_button()
            
            # Iniciar loop de procesamiento
            self.process_video()
            
            messagebox.showinfo("Sistema Iniciado", 
                              "Sistema de detección iniciado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar sistema: {str(e)}")
            self.stop_system()
    
    def stop_system(self):
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Detener threads
        if self.detection_thread:
            self.detection_thread.stop()
            
        if self.classifier_thread:
            self.classifier_thread.stop()
        
        # Liberar cámara
        if self.camera:
            self.camera.release()
        
        # Actualizar UI
        self.control_panel.enable_start_button()
        self.control_panel.show_camera_stopped()
        
        # Guardar estadísticas finales
        if self.save_stats.get():
            self.stats_logger.save()
        
        messagebox.showinfo("Sistema Detenido", 
                          "Sistema detenido correctamente")
    
    def reset_counters(self):
        self.incnt = 0
        self.outcnt = 0
        self.idcnt = []
        self.idstp = collections.defaultdict(list)
        self.trackers = []
        KalmanBoxTracker.reset_count()
        
        self.stats_logger.reset()
        self.fps_counter.reset()
        
        self.control_panel.update_live_stats(0, 0, 0, 0)
        
        messagebox.showinfo("Reiniciado", 
                          "Contadores reiniciados correctamente")
    
    def process_video(self):
        if not self.is_running:
            return
        
        try:
            # Obtener última detección
            with self.condition:
                if not self.condition.wait(timeout=0.05):
                    self.root.after(10, self.process_video)
                    return
                
                img, boxes = self.detection_thread.get_latest_detection()
            
            if img is None:
                self.root.after(10, self.process_video)
                return
            
            # Actualizar FPS
            self.fps_counter.update()
            current_fps = self.fps_counter.get_fps()
            
            # Procesar frame
            img = img.copy()
            boxes = np.array(boxes)
            
            H, W = img.shape[:2]
            line_position = int(W * self.line_position.get())  # Cambio a W para vertical
            
            # Tracking SORT
            self.perform_tracking(img, boxes, line_position, H, W)
            
            # Dibujar overlays
            self.draw_overlays(img, current_fps, line_position, H, W)
            
            # Actualizar display
            self.update_video_display(img)
            
            # Actualizar estadísticas en vivo
            self.control_panel.update_live_stats(
                current_fps, self.incnt, self.outcnt, len(self.trackers)
            )
            
            # Actualizar estadísticas del panel cada 60 frames
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 0
                
            if self._frame_count % 60 == 0:
                self.stats_panel.update_display()
            
        except Exception as e:
            print(f"Error en process_video: {e}")
        
        # Continuar loop
        self.root.after(10, self.process_video)
    
    def perform_tracking(self, img, boxes, line_position, H, W):
        # Predicción de trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Asociar detecciones con trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            boxes, trks, config.IOU_THRESHOLD
        )
        
        # Actualizar trackers matched
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                if len(d) > 0:
                    trk.update(boxes[d, :][0])
                    xmin, ymin, xmax, ymax = boxes[d, :][0]
                    cy = int((ymin + ymax) / 2)
                    
                    # Clasificar edad/sexo
                    self.classify_person(img, trk, xmin, ymin, xmax, ymax, H, W)
                    
                    # Detectar cruce de línea
                    self.detect_line_crossing(trk, cy, line_position)
                    
                    # Actualizar historial de posición
                    u, v = trk.kf.x[0], trk.kf.x[1]
                    if len(self.idstp[trk.id]) > 10:
                        self.idstp[trk.id].pop(0)
                    self.idstp[trk.id].append([u, v])
                    
                    # Dibujar tracker
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 150, 255), 2)
                    
                    label = f"{trk.id}"
                    if trk.gender:
                        label += f" {trk.gender[0]}"
                    if trk.age_range:
                        label += f" {trk.age_range}"
                    
                    cv2.putText(img, label, (int(xmin), int(ymin) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)
        
        # Crear nuevos trackers
        for i in unmatched_dets:
            if len(self.trackers) < config.MAX_TRACKERS:
                trk = KalmanBoxTracker(boxes[i, :])
                self.trackers.append(trk)
                u, v = trk.kf.x[0], trk.kf.x[1]
                self.idstp[trk.id].append([u, v])
        
        # Eliminar trackers antiguos
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > config.MAX_AGE:
                self.trackers.pop(i)
    
    def classify_person(self, img, trk, xmin, ymin, xmax, ymax, H, W):
        if (not self.enable_age_gender.get() or 
            not self.classifier_thread or 
            trk.gender is not None):
            return
        
        # Extraer región de la persona
        face_region = img[max(0, ymin):min(H, ymax), 
                         max(0, xmin):min(W, xmax)]
        
        if face_region.size > 0:
            def callback(tracker_id, gender, age_range):
                for t in self.trackers:
                    if t.id == tracker_id:
                        t.gender = gender
                        t.age_range = age_range
                        break
            
            self.classifier_thread.classify_async(trk.id, face_region, callback)
    
    def detect_line_crossing(self, trk, cy, line_position):
        if len(self.idstp[trk.id]) == 0 or trk.id in self.idcnt:
            return
        
        # Obtener posición actual del centro del tracker
        cx = int(trk.kf.x[0])  # centro x
        
        if len(self.idstp[trk.id]) > 0:
            prev_cx = self.idstp[trk.id][-1][0]  # posición x anterior
            
            # Entrada (izquierda → derecha)
            if prev_cx < line_position and cx >= line_position:
                self.incnt += 1
                self.idcnt.append(trk.id)
                print(f"ID {trk.id} - IN - {trk.gender} - {trk.age_range}")
                
                if self.save_stats.get():
                    self.stats_logger.log_entry(
                        trk.id, 'IN', 
                        gender=trk.gender, 
                        age_range=trk.age_range
                    )
            
            # Salida (derecha → izquierda)
            elif (not self.count_only_entering.get() and 
                  prev_cx >= line_position and cx < line_position):
                self.outcnt += 1
                self.idcnt.append(trk.id)
                print(f"ID {trk.id} - OUT")
                
                if self.save_stats.get():
                    self.stats_logger.log_entry(trk.id, 'OUT')
    
    def draw_overlays(self, img, fps, line_position, H, W):
        # FPS
        cv2.putText(img, f"FPS: {int(fps)}", (W - 80, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # Total
        cv2.putText(img, f"Total: {len(self.trackers)}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Línea de conteo VERTICAL
        cv2.line(img, (line_position, 0), (line_position, H), (255, 150, 0), 2)
        cv2.putText(img, f"IN: {self.incnt}", (line_position + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if not self.count_only_entering.get():
            cv2.putText(img, f"OUT: {self.outcnt}", (line_position + 5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def update_video_display(self, img):
        # Convertir BGR a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Redimensionar para el panel
        display_width = self.control_panel.video_label.winfo_width() - 20
        display_height = self.control_panel.video_label.winfo_height() - 20
        
        if display_width > 0 and display_height > 0:
            img_pil = img_pil.resize((display_width, display_height), Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.control_panel.update_video(img_tk)
    
    def on_closing(self):
        if self.is_running:
            if messagebox.askokcancel("Salir", 
                                     "¿Desea detener el sistema y salir?"):
                self.stop_system()
                self.root.destroy()
        else:
            self.root.destroy()
