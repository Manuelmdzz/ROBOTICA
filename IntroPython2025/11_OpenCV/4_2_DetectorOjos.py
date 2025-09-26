import cv2

# Cargar clasificadores Haar para rostro y ojos
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
# Iniciar captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    rostros = face_cascade.detectMultiScale(gris, 1.1, 4)

    for x, y, w, h in rostros:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gris = gris[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Detectar ojos dentro del rostro
        ojos = eye_cascade.detectMultiScale(roi_gris)
        for ex, ey, ew, eh in ojos:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("Detección de Ojos", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
