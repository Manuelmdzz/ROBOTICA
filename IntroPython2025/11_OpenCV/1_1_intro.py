import cv2

# Cargar una imagen desde un archivo
imagen = cv2.imread("Lenna.png")

# Mostrar la imagen en una ventana
cv2.imshow("Imagen Original", imagen)

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Mostrar la imagen en escala de grises
cv2.imshow("Imagen en Escala de Grises", imagen_gris)

# Guardar la imagen en escala de grises en un nuevo archivo
cv2.imwrite("imagen_gris.jpg", imagen_gris)

# Esperar a que se presione cualquier tecla
cv2.waitKey(0)

# Cerrar todas las ventanas abiertas por OpenCV
cv2.destroyAllWindows()
