#=================================================#
# stats_panel.py - v3                             #
# Configuración de estilos para la interfaz       #
# 18/01/2026                                      #
#=================================================#
from tkinter import ttk
import config


def setup_styles():
    style = ttk.Style()
    style.theme_use('clam')
    
    theme = config.THEME
    
    # Frames
    style.configure("Dark.TFrame", 
                   background=theme['bg_dark'])
    
    style.configure("Medium.TFrame", 
                   background=theme['bg_medium'])
    
    style.configure("Light.TFrame", 
                   background=theme['bg_light'])
    
    # Labels
    style.configure("Dark.TLabel",
                   background=theme['bg_dark'],
                   foreground=theme['text_color'],
                   font=('Segoe UI', 10))
    
    style.configure("Medium.TLabel",
                   background=theme['bg_medium'],
                   foreground=theme['text_color'],
                   font=('Segoe UI', 10))
    
    style.configure("Title.TLabel",
                   background=theme['bg_dark'],
                   foreground=theme['accent_blue'],
                   font=('Segoe UI', 16, 'bold'))
    
    style.configure("Subtitle.TLabel",
                   background=theme['bg_medium'],
                   foreground=theme['accent_blue'],
                   font=('Segoe UI', 14, 'bold'))
    
    style.configure("Stat.TLabel",
                   background=theme['bg_medium'],
                   foreground=theme['text_color'],
                   font=('Segoe UI', 12))
    
    style.configure("BigStat.TLabel",
                   background=theme['bg_light'],
                   foreground=theme['accent_blue'],
                   font=('Segoe UI', 24, 'bold'))
    
    style.configure("Dim.TLabel",
                   background=theme['bg_light'],
                   foreground=theme['text_dim'],
                   font=('Segoe UI', 10))
    
    # Checkbuttons
    style.configure("TCheckbutton",
                   background=theme['bg_medium'],
                   foreground=theme['text_color'],
                   font=('Segoe UI', 10))
    
    style.map("TCheckbutton",
             background=[('active', theme['bg_medium'])],
             foreground=[('active', theme['text_color'])])
    
    # Notebook (pestañas)
    style.configure("TNotebook",
                   background=theme['bg_dark'],
                   borderwidth=0)
    
    style.configure("TNotebook.Tab",
                   background=theme['bg_medium'],
                   foreground=theme['text_dim'],
                   padding=[20, 10],
                   font=('Segoe UI', 10, 'bold'))
    
    style.map("TNotebook.Tab",
             background=[('selected', theme['bg_dark'])],
             foreground=[('selected', theme['accent_blue'])],
             expand=[('selected', [1, 1, 1, 0])])
    
    # Separators
    style.configure("TSeparator",
                   background=theme['bg_light'])
    
    return style


def create_button(parent, text, command, bg_color, **kwargs):
    import tkinter as tk
    
    theme = config.THEME
    
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg_color,
        fg="white",
        font=('Segoe UI', 11, 'bold'),
        relief=tk.FLAT,
        padx=20,
        pady=10,
        cursor="hand2",
        activebackground=theme['accent_blue_hover'],
        activeforeground="white",
        **kwargs
    )
    
    # Efectos hover
    def on_enter(e):
        if btn['state'] != 'disabled':
            btn['bg'] = theme['accent_blue_hover']
    
    def on_leave(e):
        if btn['state'] != 'disabled':
            btn['bg'] = bg_color
    
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    
    return btn


def create_scale(parent, from_, to, variable, resolution=0.05):
    import tkinter as tk
    
    theme = config.THEME
    
    scale = tk.Scale(
        parent,
        from_=from_,
        to=to,
        resolution=resolution,
        orient=tk.HORIZONTAL,
        variable=variable,
        bg=theme['bg_medium'],
        fg=theme['text_color'],
        highlightthickness=0,
        troughcolor=theme['bg_dark'],
        activebackground=theme['accent_blue'],
        sliderrelief=tk.FLAT,
        font=('Segoe UI', 9)
    )
    
    return scale
