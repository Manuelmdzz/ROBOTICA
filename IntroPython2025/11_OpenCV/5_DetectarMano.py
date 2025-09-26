import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Definir el rango de color de la piel en HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Extraer la máscara de la imagen ROI
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Aplicar filtros
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    try:
        # Encontrar el contorno más grande (mano)
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(roi, [contour], -1, (0, 255, 0), 3)
    except:
        pass

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
