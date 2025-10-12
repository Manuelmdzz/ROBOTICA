# security_detect.py
import cv2
import numpy as np
import time
import os
from deepface import DeepFace
import tensorflow as tf

# --- CONFIG ---
REFERENCE_IMG = r"ManuelMendoza.jpg"  # archivo de referencia, en la misma carpeta
CAMERA_ID = 0  # 0 = webcam por defecto
DETECTOR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
BACKEND = "opencv"   # backend para detección rápida (usado solo si se quisiera DeepFace detect)
MODEL_NAME = "Facenet"  # modelo para embeddings (puedes cambiar a "VGG-Face", "ArcFace" etc.)
METRIC = "cosine"  # métrica de distancia
THRESHOLD = 0.35   # umbral de similitud (menor=mas estricto). Ajusta si hace falsos positivos/negativos.
SAVE_FOLDER = "detections"  # carpeta donde se guardan snapshots

# --- GPU / TensorFlow setup (opcional: limita crecimiento de memoria) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] GPUs detectadas: {len(gpus)} -> TensorFlow usará la GPU.")
    except Exception as e:
        print("[WARN] Error configurando GPU:", e)
else:
    print("[INFO] No se detectaron GPUs. TensorFlow usará CPU.")

# --- Preparar carpeta de guardado ---
os.makedirs(SAVE_FOLDER, exist_ok=True)

# --- Verificar imagen de referencia existe ---
if not os.path.exists(REFERENCE_IMG):
    raise FileNotFoundError(f"No se encontró la imagen de referencia: {REFERENCE_IMG}")

# --- Cargar cascade detector para detección rápida (CPU) ---
face_cascade = cv2.CascadeClassifier(DETECTOR_CASCADE)
if face_cascade.empty():
    raise RuntimeError("No se pudo cargar Haar cascade. Verifica la ruta en opencv.")

# --- Calcular embedding de la imagen de referencia (DeepFace) ---
print("[INFO] Calculando embedding de referencia...")
ref_embedding = DeepFace.represent(img_path=REFERENCE_IMG, model_name=MODEL_NAME, enforce_detection=True)[0]["embedding"]
ref_embedding = np.array(ref_embedding, dtype=np.float32)
print("[INFO] Embedding de referencia calculado.")

# --- Función de similitud (cosine) ---
def cosine_similarity(a, b):
    # devuelve 1 - cosine_distance (mayor = mas similar)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def cosine_distance(a, b):
    return 1.0 - cosine_similarity(a, b)

# --- Abrir webcam ---
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara. Verifica que esté conectada y que CAMERA_ID sea correcto.")

print("[INFO] Cámara abierta. Presiona 'q' para salir.")

frame_count = 0
last_detect_time = 0
DEBOUNCE_SECONDS = 2.0  # evita guardados continuos si la detección se mantiene

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame no leído desde la cámara.")
            time.sleep(0.1)
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar caras (rápido con Haar cascade)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        found_match = False
        for (x, y, w, h) in faces:
            # ampliar un poco la caja para mejorar recorte
            pad = int(0.15 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            face_img = frame[y1:y2, x1:x2]

            try:
                # calcular embedding del rostro detectado
                face_emb = DeepFace.represent(img_path = face_img, detector_backend = "opencv",
                                              model_name = MODEL_NAME, enforce_detection = True)[0]["embedding"]
                face_emb = np.array(face_emb, dtype=np.float32)
            except Exception as e:
                # si no se pudo representar (ej. cara muy pequeña), omitir
                # print("[DEBUG] Represent error:", e)
                continue

            # comparar con la referencia usando distancia coseno
            dist = cosine_distance(ref_embedding, face_emb)
            # dist cercano a 0 => similar. Ajustamos la lógica al umbral.
            if dist <= THRESHOLD:
                found_match = True
                # dibujar caja y texto
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "Manuel (detected)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # evitar guardar demasiadas imágenes seguidas si la persona permanece
                now = time.time()
                if now - last_detect_time > DEBOUNCE_SECONDS:
                    last_detect_time = now
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SAVE_FOLDER, f"manuel_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"[ALERTA] Manuel detectado. Snapshot guardado en: {filename} (dist={dist:.4f})")
                else:
                    # solo loguear
                    print(f"[INFO] Manuel presente (debounce). dist={dist:.4f}")

            else:
                # si quieres ver las distancias, descomenta:
                # print(f"[DEBUG] Distancia a referencia: {dist:.4f}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 1)
                cv2.putText(frame, f"dist:{dist:.3f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # mostrar el frame (opcional)
        cv2.imshow("SecurityCam - press q to quit", frame)

        # tecla para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Programa finalizado.")

