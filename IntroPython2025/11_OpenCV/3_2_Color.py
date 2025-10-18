import cv2
import numpy as np

# Variables globales para almacenar puntos y estado del mouse
drawing = False  # Verdadero cuando el ratón está siendo arrastrado
ix, iy = -1, -1  # Coordenadas iniciales de la selección
fx, fy = -1, -1  # Coordenadas finales de la selección
roi_defined = False  # Indica si la ROI ha sido definida


# Función de callback del mouse
def dibujar_rectangulo(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        roi_defined = True


def calcular_color_promedio(frame):
    global ix, iy, fx, fy
    # Asegurarse de que las coordenadas están dentro de la imagen
    x_min = max(min(ix, fx), 0)
    y_min = max(min(iy, fy), 0)
    x_max = min(max(ix, fx), frame.shape[1] - 1)
    y_max = min(max(iy, fy), frame.shape[0] - 1)
    # Extraer la región de interés (ROI)
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        print("Región seleccionada vacía. Por favor, selecciona una región válida.")
        return None
    # Calcular el color promedio en el espacio de color BGR
    avg_color_per_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = avg_color.astype(np.uint8)
    return avg_color


def main():
    global drawing, roi_defined, ix, iy, fx, fy

    cap = cv2.VideoCapture(0)  # Inicia la captura de video desde la cámara
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        return

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", dibujar_rectangulo)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        # Si el usuario está dibujando, dibujar el rectángulo actual
        if drawing:
            cv2.rectangle(frame, (ix, iy), (fx, fy), (0, 255, 0), 2)
        # Si la ROI ha sido definida, dibujar el rectángulo y calcular el color promedio
        elif roi_defined:
            cv2.rectangle(frame, (ix, iy), (fx, fy), (0, 255, 0), 2)
            avg_color = calcular_color_promedio(frame)
            if avg_color is not None:
                # Mostrar el color promedio en una ventana
                color_image = np.zeros((100, 100, 3), np.uint8)
                color_image[:] = avg_color
                cv2.imshow("Color Promedio", color_image)
                # Imprimir el color promedio en consola
                print(f"Color promedio (BGR): {avg_color}")

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF
        # Presiona 'r' para reiniciar la selección
        if key == ord("r"):
            roi_defined = False
            drawing = False
            ix, iy, fx, fy = -1, -1, -1, -1
            cv2.destroyWindow("Color Promedio")
        # Presiona 'q' para salir
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
