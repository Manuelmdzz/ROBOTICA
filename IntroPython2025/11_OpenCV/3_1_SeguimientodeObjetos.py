import cv2
import numpy as np

# Iniciar captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame al espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir el rango de color a detectar (ejemplo: azul)
    lower_color = np.array([100, 150, 0])
    upper_color = np.array([140, 255, 255])

    # Definir el rango de color a detectar (ejemplo: amarillo)
    # lower_color = np.array([53, 220, 197]) - 30  # [100, 150, 0])
    # upper_color = np.array([53, 220, 197]) + 20

    # Crear una máscara con los rangos definidos
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Mostrar las imágenes
    cv2.imshow("Frame", frame)
    cv2.imshow("Mascara", mask)
    cv2.imshow("Resultado", result)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
