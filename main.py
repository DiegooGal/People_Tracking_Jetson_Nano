#!/usr/bin/env python3

#=================================================#
# main.py - v7                                    #
# Sistema de Conteo de Personas con Jetson Nano   #
# Panel de Administración con GUI                 #
# Punto de entrada principal de la aplicación     #
# 20/01/2026                                      #
#=================================================#

import tkinter as tk
import sys
import os

# Añadir directorio actual al path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import MainWindow


def main():
    """Función principal"""
    print("=" * 60)
    print("Sistema de Conteo de Personas - Panel de Administración")
    print("=" * 60)
    print()
    
    # Crear ventana principal
    root = tk.Tk()
    
    # Crear aplicación
    app = MainWindow(root)
    
    # Iniciar loop de eventos
    print("Interfaz gráfica iniciada")
    print("Presiona 'Iniciar Sistema' para comenzar")
    print()
    
    root.mainloop()
    
    print()
    print("Aplicación cerrada")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrumpido por usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
