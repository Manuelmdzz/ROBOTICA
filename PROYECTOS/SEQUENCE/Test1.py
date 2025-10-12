import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"✅ Cámara detectada en índice: {i}")
        cap.release()
    else:
        print(f"❌ No hay cámara en índice: {i}")
