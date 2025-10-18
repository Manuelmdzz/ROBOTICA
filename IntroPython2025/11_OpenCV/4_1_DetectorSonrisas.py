import cv2

# Cargar el clasificador en cascada de Haar para la detección de rostros y sonrisas
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Leer y mostrar frames de la cámara en un bucle
while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Si no se pudo capturar un frame, salir del bucle
    if not ret:
        print(
            "No se pudo recibir frame (el stream puede haber terminado). Saliendo ..."
        )
        break

    # Convertir el frame a escala de grises (la detección es más eficiente en escala de grises)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Iterar sobre cada rostro detectado
    for x, y, w, h in faces:
        # Dibujar un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Obtener la región del rostro para la detección de sonrisas
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Detectar sonrisas en la región del rostro
        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25)
        )

        # Dibujar un rectángulo alrededor de cada sonrisa detectada
        for sx, sy, sw, sh in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

    # Mostrar el frame con los rostros y sonrisas detectados
    cv2.imshow("Detección de Sonrisas", frame)

    # Esperar 1 ms para ver si se presiona la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la captura de video cuando se haya terminado
cap.release()

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()
