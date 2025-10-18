# Práctica de Visión por Computadora para Robótica con scikit-image

## Objetivo

Desarrollar habilidades en procesamiento de imágenes aplicadas a robótica usando scikit-image, simulando tareas típicas que un robot necesita realizar.

## Configuración Inicial

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology, segmentation
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny, corner_harris, corner_peaks
from skimage.transform import hough_line, hough_line_peaks, hough_circle
from skimage.util import random_noise
import cv2  # Para algunas funciones complementarias
```

---

## Ejercicio 1: Detección de Objetos por Color (Clasificación de Piezas)

**Escenario**: Un robot debe clasificar piezas de diferentes colores en una banda transportadora.

**Tareas**:

1. Carga una imagen con objetos de diferentes colores
2. Convierte a espacio de color HSV
3. Crea máscaras para detectar objetos rojos, azules y verdes
4. Cuenta el número de objetos de cada color
5. Calcula el centroide de cada objeto

```python
def detectar_objetos_por_color(imagen):
    """
    Detecta y clasifica objetos por color
    """
    # Tu código aquí
    pass

# Imagen de prueba (puedes usar una imagen sintética o real)
# imagen = io.imread('piezas_colores.jpg')
```

**Criterios de evaluación**:

- Precisión en la detección (>90%)
- Robustez ante variaciones de iluminación
- Cálculo correcto de centroides

---

## Ejercicio 2: Navegación por Líneas (Seguimiento de Trayectoria)

**Escenario**: Un robot móvil debe seguir una línea en el suelo para navegar.

**Tareas**:

1. Detecta líneas usando la transformada de Hough
2. Calcula el ángulo de la línea principal
3. Determina si el robot debe girar izquierda/derecha
4. Estima la distancia al centro de la línea

```python
def seguir_linea(imagen):
    """
    Procesa imagen para seguimiento de línea
    Retorna: ángulo_giro, distancia_centro
    """
    # Preprocesamiento
    gray = rgb2gray(imagen)

    # Detección de bordes
    edges = canny(gray, sigma=2, low_threshold=0.1, high_threshold=0.2)

    # Tu código para Hough Transform aquí

    return angulo_giro, distancia_centro
```

**Reto adicional**: Maneja intersecciones y bifurcaciones.

---

## Ejercicio 3: Detección y Medición de Objetos (Control de Calidad)

**Escenario**: Un robot industrial debe medir dimensiones de piezas manufacturadas.

**Tareas**:

1. Segmenta objetos del fondo
2. Calcula área, perímetro y dimensiones principales
3. Detecta defectos (agujeros, irregularidades)
4. Clasifica piezas como "aprobadas" o "defectuosas"

```python
def control_calidad(imagen, tolerancia_area=0.1):
    """
    Analiza piezas para control de calidad
    """
    # Segmentación
    gray = rgb2gray(imagen)
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh

    # Análisis morfológico
    cleaned = morphology.remove_small_objects(binary, min_size=100)

    # Tu código para mediciones aquí

    return resultados_medicion
```

**Métricas a calcular**:

- Área en píxeles y mm²
- Relación aspecto (largo/ancho)
- Circularidad: 4π×área/perímetro²
- Número de agujeros

---

## Ejercicio 4: Reconocimiento de Formas Geométricas

**Escenario**: Un robot debe clasificar y manipular objetos según su forma.

**Tareas**:

1. Detecta contornos de objetos
2. Clasifica formas: círculo, cuadrado, triángulo, rectángulo
3. Calcula orientación para agarre robótico
4. Determina puntos de agarre óptimos

```python
def clasificar_formas(imagen):
    """
    Clasifica objetos por su forma geométrica
    """
    # Detección de contornos
    gray = rgb2gray(imagen)
    edges = canny(gray, sigma=1)

    # Análisis de forma
    contours = measure.find_contours(edges, 0.5)

    formas_detectadas = []

    for contour in contours:
        # Tu código para clasificación aquí
        # Usa aproximación poligonal, análisis de curvatura, etc.
        pass

    return formas_detectadas
```

**Algoritmos sugeridos**:

- Aproximación de Douglas-Peucker
- Análisis de momentos de Hu
- Descriptores de Fourier

---
