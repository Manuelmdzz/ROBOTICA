import cv2

# Cargar el clasificador en cascada de Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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

    # Dibujar un rectángulo alrededor de cada rostro detectado
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar el frame con los rostros detectados
    cv2.imshow("Detección de Rostros", frame)

    # Esperar 1 ms para ver si se presiona la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la captura de video cuando se haya terminado
cap.release()

# Cerrar todas las ventanas de OpenCV
cv2
