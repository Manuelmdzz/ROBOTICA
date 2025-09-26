import cv2

# Cargar el clasificador en cascada de Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Cargar una imagen
imagen = cv2.imread("Lenna.png")

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectar rostros en la imagen
rostros = face_cascade.detectMultiScale(
    imagen_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

# Dibujar un rectángulo alrededor de cada rostro detectado
for x, y, w, h in rostros:
    cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Mostrar la imagen con los rostros detectados
cv2.imshow("Detección de Rostros", imagen)

# Esperar a que se presione cualquier tecla
cv2.waitKey(0)

# Cerrar todas las ventanas abiertas por OpenCV
cv2.destroyAllWindows()
