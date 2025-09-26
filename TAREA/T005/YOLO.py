import cv2
import numpy as np
from ultralytics import YOLO

# === Config ===
# Usa un modelo de SEGMENTACIÓN (termina en -seg). Si no puedes descargar, cambia a "yolov8n-seg.pt".
MODEL_WEIGHTS = "yolo11n-seg.pt"
CONF_THRESH = 0.25
CAMERA_INDEX = 0       # Cambia a 1/2 si tienes varias cámaras
SHOW_FPS = True

# Si no hay máscaras (p.ej. si usas un modelo sin -seg), usa cajas como respaldo
USE_BOX_FALLBACK = True

def open_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def blackout_persons(frame, results):
    """
    Pinta de negro las regiones de 'person' usando máscaras; si no hay máscaras, usa cajas (fallback).
    Corrige desajuste de tamaños redimensionando la máscara al tamaño del frame.
    """
    import numpy as np
    import cv2

    out = frame.copy()
    r = results[0]

    # --- Con máscaras (segmentación) ---
    if hasattr(r, "masks") and r.masks is not None and r.masks.data is not None:
        masks = r.masks.data  # tensor [N, h, w]
        if hasattr(masks, "cpu"):
            masks = masks.cpu().numpy()
        else:
            masks = np.asarray(masks)

        # clases
        classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
        names = getattr(r, "names", None)
        if names is None:
            # intenta obtener de results (según versión de ultralytics)
            names = getattr(results, "names", None)
        if names is None:
            # fallback
            names = {0: "person"}

        H, W = out.shape[:2]

        for i, m in enumerate(masks):
            cls_id = classes[i] if i < len(classes) else -1
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

            if cls_name == "person" or cls_id == 0:  # COCO: person=0
                # m es float en [0..1] o [0..255]; conviértela a bool con umbral
                m_uint8 = (m > 0.5).astype("uint8") * 255
                # --- ¡clave!: redimensionar la máscara al tamaño del frame ---
                m_resized = cv2.resize(m_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bool = m_resized.astype(bool)
                out[mask_bool] = (0, 0, 0)
        return out

    # --- Respaldo con cajas ---
    if r.boxes is not None:
        boxes = r.boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        names = getattr(r, "names", None) or getattr(results, "names", None) or {0: "person"}

        for i, b in enumerate(boxes.xyxy.cpu().numpy()):
            cls_id = classes[i]
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            if cls_name == "person" or cls_id == 0:
                x1, y1, x2, y2 = b.astype(int)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(out.shape[1]-1, x2), min(out.shape[0]-1, y2)
                out[y1:y2, x1:x2] = (0, 0, 0)
        return out

    return out

def draw_fps(img, fps):
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def main():
    print("[INFO] Cargando modelo de segmentación...")
    model = YOLO(MODEL_WEIGHTS)  # descarga si no existe
    cap = open_camera(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {CAMERA_INDEX}. Cierra otras apps o cambia el índice.")
        return

    print("[INFO] Presiona 'q' para salir. 's' para guardar un frame.")
    import time
    t0 = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame no válido.")
            break

        # Inferencia (usa GPU si está disponible automáticamente)
        results = model.predict(
            frame,
            conf=CONF_THRESH,
            imgsz=640,
            verbose=False
        )

        # Pinta de negro a las personas
        masked = blackout_persons(frame, results)

        # FPS simple
        frame_count += 1
        if frame_count >= 10:
            t1 = time.time()
            fps = frame_count / (t1 - t0)
            t0 = t1
            frame_count = 0

        if SHOW_FPS:
            draw_fps(masked, fps)

        cv2.imshow("Personas en negro (YOLO seg)", masked)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"frame_masked_{int(time.time())}.jpg"
            cv2.imwrite(fname, masked)
            print(f"[INFO] Guardado {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
