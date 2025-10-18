import cv2
import time
import numpy as np
from ultralytics import YOLO

# ================== Config ==================
MODEL_WEIGHTS = "yolo11n.pt"    # también: "yolov8n.pt"
CAMERA_INDEX  = 0               # cambia a 1/2 si tienes varias cámaras
CONF_THRESH   = 0.35
IOU_THRESH    = 0.5
IMG_SIZE      = 640
TRACKER_CFG   = "bytetrack.yaml"  # integrado en ultralytics
PERSON_CLASS  = 0               # COCO: 'person' = 0
SHOW_FPS      = True
# ============================================

def open_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def draw_box_id(img, box, track_id, name="person", conf=None, color=(0, 200, 255)):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{name} ID:{track_id}"
    if conf is not None:
        label += f" {conf:.2f}"
    cv2.putText(img, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def main():
    print("[INFO] Cargando modelo...")
    model = YOLO(MODEL_WEIGHTS)

    cap = open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {CAMERA_INDEX}.")
        return

    print("[INFO] Presiona 'q' para salir. 's' para guardar un frame.")
    t0, frames, fps = time.time(), 0, 0.0

    # persist=True mantiene los IDs entre frames dentro de esta ejecución
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame no válido.")
            break

        results = model.track(
            frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            classes=[PERSON_CLASS],   # Solo personas
            tracker=TRACKER_CFG,
            persist=True,
            verbose=False
        )

        vis = frame.copy()
        r = results[0]
        boxes = getattr(r, "boxes", None)

        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
            ids  = boxes.id
            ids  = ids.cpu().numpy().astype(int) if ids is not None else [None]*len(xyxy)
            conf = boxes.conf.cpu().numpy() if boxes.conf is not None else [None]*len(xyxy)

            names = getattr(r, "names", None) or getattr(results, "names", None) or {0: "person"}

            for i, box in enumerate(xyxy):
                if cls[i] != PERSON_CLASS:
                    continue
                track_id = ids[i] if ids is not None else None
                # Si por alguna razón no hay ID (pérdida temporal), coloca -1
                track_id = int(track_id) if track_id is not None else -1
                name = names.get(PERSON_CLASS, "person") if isinstance(names, dict) else "person"
                draw_box_id(vis, box, track_id, name=name, conf=(conf[i] if conf is not None else None))

        # FPS cada 10 frames
        frames += 1
        if frames >= 10:
            t1 = time.time()
            fps = frames / (t1 - t0)
            t0, frames = t1, 0

        if SHOW_FPS:
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 220, 50), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Person Tracking (IDs)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"tracked_{int(time.time())}.jpg"
            cv2.imwrite(fname, vis)
            print(f"[INFO] Guardado {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
