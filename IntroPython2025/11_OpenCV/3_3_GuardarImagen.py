import cv2
import os
from datetime import datetime


def main():
    # Nombre de la carpeta donde se guardarán las imágenes
    carpeta = "ImagenesCam"

    # Crear la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        print(f"Carpeta '{carpeta}' creada.")
    else:
        print(f"Carpeta '{carpeta}' ya existe.")

    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    contador = 0  # Contador para nombrar las imágenes

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        # cv2.putText(
        #    frame,
        #    "Presionar 'S' para guardar imagen",
        #    (50, 90),
        #    cv2.FONT_HERSHEY_SIMPLEX,
        #    1,
        #    (0, 255, 0),
        #    2,
        #    cv2.LINE_AA,
        # )

        # Mostrar el frame en una ventana
        cv2.imshow("Captura", frame)

        # Esperar por una tecla
        key = cv2.waitKey(1) & 0xFF

        # Presionar 's' para guardar la imagen
        if key == ord("s"):

            # Opción 1: Guardar con un contador
            nombre_imagen = f"Imagen_{contador}.jpg"
            contador += 1

            # Opción 2: Guardar con la fecha y hora actual
            # ahora = datetime.now()
            # nombre_imagen = ahora.strftime("%Y%m%d_%H%M%S.jpg")

            # Ruta completa para guardar la imagen
            ruta_imagen = os.path.join(carpeta, nombre_imagen)

            # Guardar la imagen
            cv2.imwrite(ruta_imagen, frame)
            print(f"Imagen guardada: {ruta_imagen}")

        # Presionar 'q' para salir
        elif key == ord("q"):
            break

    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
