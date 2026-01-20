#=================================================#
# stats_panel.py - v3                             #
# Panel de estadísticas del sistema               #
# 18/01/2026                                      #
#=================================================#
import tkinter as tk
from tkinter import ttk
import config
from gui.styles import create_button


class StatsPanel:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.theme = config.THEME
        
        self.create_ui()
    
    def create_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.parent, style="Dark.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        ttk.Label(main_frame, text="Estadísticas Acumuladas", 
                 style="Title.TLabel").pack(pady=(0, 20))
        
        # Estadísticas generales
        self.create_general_stats(main_frame)
        
        # Estadísticas demográficas
        self.create_demographic_stats(main_frame)
        
        # Botón de actualización
        update_btn = create_button(
            main_frame,
            text="↻ Actualizar Estadísticas",
            command=self.update_display,
            bg_color=self.theme['accent_blue']
        )
        update_btn.pack(pady=10)
    
    def create_general_stats(self, parent):
        general_frame = ttk.Frame(parent, style="Medium.TFrame")
        general_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(general_frame, text="Estadísticas Generales",
                 style="Subtitle.TLabel").pack(pady=10)
        
        # Grid de estadísticas
        stats_grid = ttk.Frame(general_frame, style="Medium.TFrame")
        stats_grid.pack(fill=tk.X, padx=20, pady=10)
        
        # Configurar columnas (4 columnas ahora)
        for i in range(4):
            stats_grid.columnconfigure(i, weight=1)
        
        # Crear cajas de estadísticas
        self.total_in_stat = self.create_stat_box(
            stats_grid, "Entradas", "0", 0, 0)
        
        self.total_out_stat = self.create_stat_box(
            stats_grid, "Salidas", "0", 0, 1)
        
        self.total_net_stat = self.create_stat_box(
            stats_grid, "Neto", "0", 0, 2)
        
        self.total_crossings_stat = self.create_stat_box(
            stats_grid, "Total Cruces", "0", 0, 3)
    
    def create_stat_box(self, parent, title, value, row, col):
        frame = ttk.Frame(parent, style="Light.TFrame")
        frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        # Título
        ttk.Label(frame, text=title, style="Dim.TLabel").pack(pady=(10, 5))
        
        # Valor
        value_label = ttk.Label(frame, text=value, style="BigStat.TLabel")
        value_label.pack(pady=(0, 10))
        
        return value_label
    
    def create_demographic_stats(self, parent):
        demo_frame = ttk.Frame(parent, style="Medium.TFrame")
        demo_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(demo_frame, text="Demografía y Tráfico",
                 style="Subtitle.TLabel").pack(pady=10)
        
        # Área de texto con scroll
        text_frame = ttk.Frame(demo_frame, style="Medium.TFrame")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Scrollbar
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget
        self.demo_text = tk.Text(
            text_frame,
            height=20,
            bg=self.theme['bg_light'],
            fg=self.theme['text_color'],
            font=('Consolas', 10),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            yscrollcommand=scrollbar.set,
            wrap=tk.WORD
        )
        self.demo_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.demo_text.yview)
    
    def update_display(self):
        stats_summary = self.app.stats_logger.get_summary()
        
        # Actualizar estadísticas generales
        self.total_in_stat.config(text=str(stats_summary['total_in']))
        self.total_out_stat.config(text=str(stats_summary['total_out']))
        self.total_net_stat.config(text=str(stats_summary['net']))
        
        # Total de cruces (entradas + salidas)
        total_crossings = stats_summary['total_in'] + stats_summary['total_out']
        self.total_crossings_stat.config(text=str(total_crossings))
        
        # Actualizar texto demográfico
        self.update_demographic_text(stats_summary)
    
    def update_demographic_text(self, summary):
        self.demo_text.config(state=tk.NORMAL)  # Habilitar edición
        self.demo_text.delete('1.0', tk.END)
        
        text = ""
        
        # NOTA: Solo mostramos demografía de entradas
        text += "DEMOGRAFIA (Solo personas que entraron)\n"
        text += "=" * 60 + "\n\n"
        
        # Sección de género
        text += "GENERO\n"
        text += "-" * 60 + "\n"
        
        total_entries = summary['total_in']
        gender_stats = summary['demographics']['gender']
        
        if gender_stats and total_entries > 0:
            # Filtrar "None" y "Desconocido"
            valid_genders = {k: v for k, v in gender_stats.items() 
                           if k and k != 'None' and k != 'Desconocido' and v > 0}
            
            if valid_genders:
                for gender, count in valid_genders.items():
                    percentage = (count / total_entries) * 100
                    bar = "█" * int(percentage / 2)
                    text += f"  {gender:<15} {count:>5} ({percentage:>5.1f}%) {bar}\n"
            else:
                text += "  No hay datos clasificados aun\n"
        else:
            text += "  No hay datos disponibles\n"
        
        text += "\n"
        
        # Sección de edad
        text += "RANGOS DE EDAD\n"
        text += "-" * 60 + "\n"
        
        age_stats = summary['demographics']['age_ranges']
        
        if age_stats and total_entries > 0:
            # Filtrar "None" y "Desconocido"
            valid_ages = {k: v for k, v in age_stats.items() 
                         if k and k != 'None' and k != 'Desconocido' and v > 0}
            
            if valid_ages:
                for age_range, count in sorted(valid_ages.items()):
                    percentage = (count / total_entries) * 100
                    bar = "█" * int(percentage / 2)
                    text += f"  {age_range:<15} {count:>5} ({percentage:>5.1f}%) {bar}\n"
            else:
                text += "  No hay datos clasificados aun\n"
        else:
            text += "  No hay datos disponibles\n"
        
        text += "\n"
        text += "=" * 60 + "\n\n"
        
        # Sección de tráfico por hora
        text += "TRAFICO POR HORA (Entradas + Salidas)\n"
        text += "=" * 60 + "\n"
        
        hourly_traffic = summary['hourly_traffic']
        
        if hourly_traffic:
            for hour, data in sorted(hourly_traffic.items()):
                total = data['in'] + data.get('out', 0)
                text += f"\n  {hour}\n"
                text += f"     Total: {total} personas\n"
                text += f"     -> IN:  {data['in']}\n"
                text += f"     <- OUT: {data.get('out', 0)}\n"
        else:
            text += "  No hay datos disponibles\n"
        
        text += "\n"
        
        # Hora pico
        if summary['peak_hour']:
            text += "HORA PICO\n"
            text += "=" * 60 + "\n"
            text += f"  * {summary['peak_hour']}\n"
            text += f"     {summary['peak_traffic']} personas\n"
        
        self.demo_text.insert('1.0', text)
        
        # Hacer el texto de solo lectura
        self.demo_text.config(state=tk.DISABLED)
