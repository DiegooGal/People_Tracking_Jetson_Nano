#=================================================#
# control_panel.py - v2                           #
# Panel de control y visualización de cámara      #
# 18/01/2026                                      #
#=================================================#
import tkinter as tk
from tkinter import ttk
import PIL
from PIL import Image
import PIL.ImageTk as ImageTk
import config
from gui.styles import create_button, create_scale


class ControlPanel:
    #Panel de control izquierdo y vista de cámara derecha
    
    def __init__(self, parent, app):
        self.app = app
        self.theme = config.THEME
        
        self.create_ui()
    
    def create_ui(self):
        # Frame izquierdo: Configuración
        self.left_frame = ttk.Frame(self.parent, style="Medium.TFrame", width=350)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        self.left_frame.pack_propagate(False)
        
        # Título
        ttk.Label(self.left_frame, text="Configuración", 
                  style="Title.TLabel").pack(pady=(10, 20))
        
        # Sección de opciones
        self.create_options_section()
        
        # Separador
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=15, pady=20)
        
        # Sección de sliders
        self.create_sliders_section()
        
        # Separador
        ttk.Separator(self.left_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=15, pady=20)
        
        # Botones de control
        self.create_control_buttons()
        
        # Estadísticas en vivo
        self.create_live_stats()
        
        # Frame derecho: Video
        self.create_video_panel()
    
    def create_options_section(self):
        options_frame = ttk.Frame(self.left_frame, style="Medium.TFrame")
        options_frame.pack(fill=tk.X, padx=15, pady=5)
        
        ttk.Checkbutton(
            options_frame,
            text="Guardar Estadísticas",
            variable=self.app.save_stats,
            style="TCheckbutton"
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            options_frame,
            text="Seguimiento por Horas",
            variable=self.app.track_hourly,
            style="TCheckbutton"
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            options_frame,
            text="Solo Contar Entradas",
            variable=self.app.count_only_entering,
            style="TCheckbutton"
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            options_frame,
            text="Clasificación Edad/Sexo",
            variable=self.app.enable_age_gender,
            style="TCheckbutton"
        ).pack(anchor=tk.W, pady=5)
    
    def create_sliders_section(self):
        slider_frame = ttk.Frame(self.left_frame, style="Medium.TFrame")
        slider_frame.pack(fill=tk.X, padx=15, pady=5)
        
        # Posición de línea
        ttk.Label(slider_frame, text="Posición de Línea (Vertical)", 
                 style="Medium.TLabel").pack(anchor=tk.W, pady=(10, 5))
        
        self.line_scale = create_scale(
            slider_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.app.line_position,
            resolution=0.05
        )
        self.line_scale.pack(fill=tk.X, pady=5)
        
        # Confianza de detección
        ttk.Label(slider_frame, text="Confianza de Detección", 
                 style="Medium.TLabel").pack(anchor=tk.W, pady=(10, 5))
        
        self.conf_scale = create_scale(
            slider_frame,
            from_=0.3,
            to=0.9,
            variable=self.app.confidence,
            resolution=0.05
        )
        self.conf_scale.pack(fill=tk.X, pady=5)
    
    def create_control_buttons(self):
        btn_frame = ttk.Frame(self.left_frame, style="Medium.TFrame")
        btn_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Botón iniciar
        self.start_btn = create_button(
            btn_frame,
            text="▶ Iniciar Sistema",
            command=self.app.start_system,
            bg_color=self.theme['accent_blue']
        )
        self.start_btn.pack(fill=tk.X, pady=5)
        
        # Botón detener
        self.stop_btn = create_button(
            btn_frame,
            text="■ Detener Sistema",
            command=self.app.stop_system,
            bg_color=self.theme['accent_red'],
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        # Botón reiniciar
        self.reset_btn = create_button(
            btn_frame,
            text="↻ Reiniciar Contadores",
            command=self.app.reset_counters,
            bg_color=self.theme['accent_gray']
        )
        self.reset_btn.pack(fill=tk.X, pady=5)
    
    def create_live_stats(self):
        stats_frame = ttk.Frame(self.left_frame, style="Light.TFrame")
        stats_frame.pack(fill=tk.X, padx=15, pady=10)
        
        ttk.Label(stats_frame, text="Estado en Vivo",
                 style="Subtitle.TLabel").pack(pady=10)
        
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0", 
                                   style="Stat.TLabel")
        self.fps_label.pack(pady=5)
        
        self.in_label = ttk.Label(stats_frame, text="Entradas: 0", 
                                  style="Stat.TLabel")
        self.in_label.pack(pady=5)
        
        self.out_label = ttk.Label(stats_frame, text="Salidas: 0", 
                                   style="Stat.TLabel")
        self.out_label.pack(pady=5)
        
        self.total_label = ttk.Label(stats_frame, text="Total Actual: 0", 
                                     style="Stat.TLabel")
        self.total_label.pack(pady=5)
    
    def create_video_panel(self):
        right_frame = ttk.Frame(self.parent, style="Dark.TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, 
                        padx=(0, 10), pady=10)
        
        ttk.Label(right_frame, text="Vista de Cámara", 
                 style="Title.TLabel").pack(pady=10)
        
        self.video_label = tk.Label(
            right_frame,
            bg=self.theme['bg_medium'],
            text="Cámara detenida\n\nPresiona 'Iniciar Sistema' para comenzar",
            fg=self.theme['text_dim'],
            font=('Segoe UI', 14)
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def update_live_stats(self, fps, in_count, out_count, total):
        self.fps_label.config(text=f"FPS: {int(fps)}")
        self.in_label.config(text=f"Entradas: {in_count}")
        self.out_label.config(text=f"Salidas: {out_count}")
        self.total_label.config(text=f"Total Actual: {total}")
    
    def update_video(self, img_tk):
        self.video_label.config(image=img_tk, text='')
        self.video_label.image = img_tk
    
    def show_camera_stopped(self):
        self.video_label.config(
            image='',
            text="Cámara detenida\n\nPresiona 'Iniciar Sistema' para comenzar"
        )
    
    def enable_start_button(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def enable_stop_button(self):
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
