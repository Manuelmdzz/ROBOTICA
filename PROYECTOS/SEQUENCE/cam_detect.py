# cam_detect_warped.py
from pathlib import Path
from ultralytics import YOLO
import torch, cv2, time, os
import numpy as np

# === CONFIG ===
BASE_RUNS_DIR = r"C:\UP\PYTHON\ROBOTICS\PROYECTOS\SEQUENCE\runs_sequence_2"
RUN_PATTERN   = "Sequence_yolo11*"
IMGSZ         = (800, 608)
CONF, IOU     = 0.5, 0.45 # Un poco más de confianza puede ayudar

# --- Configuración de Cámara ---
USE_ESP32_CAM = False
ESP32_STREAM_URL = "http://192.168.137.123:81/stream" # <-- USA LA IP DEL HOTSPOT

WINDOW_NAME   = "Deteccion Original"
WARPED_WINDOW_NAME = "Tablero Corregido"
SAVE_DIR      = r"C:\UP\PYTHON\ROBOTICS\PROYECTOS\SEQUENCE\runs_sequence_2\pred_cam"

def latest_best_pt(base_dir: str, pattern: str) -> Path:
    # ... (sin cambios en esta función)
    base = Path(base_dir)
    cands = [p for p in base.glob(pattern) if (p / "weights" / "best.pt").exists()]
    if not cands:
        raise FileNotFoundError(f"No encontré 'best.pt' bajo {base_dir}\\{pattern}.")
    newest = max(cands, key=lambda p: (p / "weights" / "best.pt").stat().st_mtime)
    return newest / "weights" / "best.pt"

def order_points(pts):
    # Ordena los 4 puntos: arriba-izq, arriba-der, abajo-der, abajo-izq
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def main():
    ckpt = latest_best_pt(BASE_RUNS_DIR, RUN_PATTERN)
    print(f"Usando checkpoint: {ckpt}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(str(ckpt))

    cap = cv2.VideoCapture(ESP32_STREAM_URL if USE_ESP32_CAM else 0)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la fuente de video.")

    prev_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Frame no disponible, reintentando...")
            time.sleep(1)
            continue
        
        frame = cv2.resize(frame, IMGSZ)

        results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF, iou=IOU, device=device, verbose=False)
        annotated_frame = results[0].plot()

        # --- LÓGICA DE CORRECCIÓN DE PERSPECTIVA ---
        board_bbox = None
        BOARD_CLASS_ID = 1 # Asume que la clase 'Board' es 0. ¡Verifícalo!

        for r in results:
            for box in r.boxes:
                if int(box.cls) == BOARD_CLASS_ID and box.conf > CONF:
                    board_bbox = box.xyxy[0].cpu().numpy().astype(int)
                    break
            if board_bbox is not None:
                break

        if board_bbox is not None:
            # 1. Recorta el bounding box del frame original para aislar el tablero
            x1, y1, x2, y2 = board_bbox
            board_roi = frame[y1:y2, x1:x2]

            # 2. Encuentra las 4 esquinas del tablero dentro del ROI
            gray_roi = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Encuentra el contorno más grande
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Aproxima el contorno a un polígono de 4 lados
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Si tenemos 4 esquinas, procedemos a la transformación
                if len(approx_corners) == 4:
                    # Convierte las coordenadas de las esquinas del ROI a coordenadas del frame completo
                    src_points = order_points(approx_corners.reshape(4, 2) + np.array([x1, y1]))
                    
                    # Dibuja círculos en las esquinas detectadas sobre la imagen original
                    for point in src_points:
                        cv2.circle(annotated_frame, tuple(point.astype(int)), 5, (0, 0, 255), -1)

                    # 3. Define el rectángulo de destino (vista cenital)
                    width_a = np.linalg.norm(src_points[2] - src_points[3])
                    width_b = np.linalg.norm(src_points[1] - src_points[0])
                    max_width = max(int(width_a), int(width_b))

                    height_a = np.linalg.norm(src_points[1] - src_points[2])
                    height_b = np.linalg.norm(src_points[0] - src_points[3])
                    max_height = max(int(height_a), int(height_b))
                    
                    # Creamos las coordenadas de destino
                    dst_points = np.array([
                        [0, 0],
                        [max_width - 1, 0],
                        [max_width - 1, max_height - 1],
                        [0, max_height - 1]], dtype="float32")

                    # 4. Calcula la matriz de transformación y aplica el warp
                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    warped_board = cv2.warpPerspective(frame, matrix, (max_width, max_height))

                    # 5. Dibuja la cuadrícula 10x10 en la imagen corregida
                    rows, cols = 10, 10
                    cell_height = max_height // rows
                    cell_width = max_width // cols
                    for i in range(1, cols):
                        cv2.line(warped_board, (i * cell_width, 0), (i * cell_width, max_height), (255, 255, 0), 1)
                    for i in range(1, rows):
                        cv2.line(warped_board, (0, i * cell_height), (max_width, i * cell_height), (255, 255, 0), 1)
                    
                    # Ahora puedes analizar 'warped_board' para detectar fichas por casilla
                    cv2.imshow(WARPED_WINDOW_NAME, warped_board)
        
        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Fin de la sesión.")

if __name__ == "__main__":
    main()