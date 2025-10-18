import cv2

# Iniciar la captura de video desde la cámara web
# El parámetro '0' se refiere a la cámara por defecto del sistema
# Si tienes más de una cámara, puedes usar '1', '2', etc., dependiendo de la cámara que desees utilizar
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

    # Mostrar el frame en una ventana llamada 'Video'
    cv2.imshow("Video", frame)

    # Esperar 1 ms para ver si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la captura de video cuando se haya terminado
cap.release()

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()
