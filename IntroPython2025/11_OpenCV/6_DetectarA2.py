import cv2
import pytesseract
from pytesseract import Output

# Configurar la ruta al ejecutable de Tesseract si es necesario
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ruta en Windows

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

    # Convertir el frame a escala de grises (la OCR es más eficiente en escala de grises)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral binario para mejorar el contraste
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Usar Tesseract para realizar OCR en el frame y obtener la información de los caracteres detectados
    d = pytesseract.image_to_data(thresh, output_type=Output.DICT)

    # Recorrer todos los elementos detectados por Tesseract
    n_boxes = len(d["level"])
    for i in range(n_boxes):
        text = d["text"][i]
        if "A" in text:
            (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
            # Dibujar un rectángulo verde alrededor de la letra "A"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame con el resultado de OCR
    cv2.imshow("Detección de la letra A", frame)

    # Esperar 1 ms para ver si se presiona la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la captura de video cuando se haya terminado
cap.release()

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()
