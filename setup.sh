#!/bin/bash
# Script de configuración del proyecto People Counter

echo "====================================="
echo "People Counter - Setup"
echo "====================================="
echo

# Crear estructura de directorios
echo "Creando estructura de directorios..."

mkdir -p core
mkdir -p gui
mkdir -p utils
mkdir -p models

# Crear archivos __init__.py
echo "Creando archivos __init__.py..."

touch core/__init__.py
touch gui/__init__.py
touch utils/__init__.py

# Verificar dependencias
echo
echo "Verificando dependencias de Python..."

python3 -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  OpenCV no encontrado"
fi

python3 -c "import pycuda" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  PyCUDA no encontrado"
fi

python3 -c "import PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Pillow no encontrado"
fi

python3 -c "import filterpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  FilterPy no encontrado"
fi

python3 -c "import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Scikit-learn no encontrado"
fi

python3 -c "import numba" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Numba no encontrado"
fi

# Verificar modelos
echo
echo "Verificando modelos de edad/sexo..."

if [ -f "age_deploy.prototxt" ]; then
    echo "✓ age_deploy.prototxt encontrado"
else
    echo "✗ age_deploy.prototxt NO encontrado"
fi

if [ -f "age_render.caffemodel" ]; then
    echo "✓ age_render.caffemodel encontrado"
else
    echo "✗ age_render.caffemodel NO encontrado"
fi

if [ -f "gender_deploy.prototxt" ]; then
    echo "✓ gender_deploy.prototxt encontrado"
else
    echo "✗ gender_deploy.prototxt NO encontrado"
fi

if [ -f "gender_render.caffemodel" ]; then
    echo "✓ gender_render.caffemodel encontrado"
else
    echo "✗ gender_render.caffemodel NO encontrado"
fi

if [ -f "mean.binaryproto" ]; then
    echo "✓ mean.binaryproto encontrado"
else
    echo "✗ mean.binaryproto NO encontrado"
fi

# Hacer main.py ejecutable
chmod +x main.py

echo
echo "====================================="
echo "Setup completado"
echo "====================================="
echo
echo "Para ejecutar la aplicación:"
echo "  python3 main.py"
echo
echo "O directamente:"
echo "  ./main.py"
echo
