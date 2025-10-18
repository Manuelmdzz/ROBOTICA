import cv2
import numpy as np
from ultralytics import YOLO
import time

# ================== Config ==================
MODEL_WEIGHTS = "yolo11n-pose.pt"  # Alternativa estable: "yolov8n-pose.pt"
CONF_THRESH   = 0.5
CAMERA_INDEX  = 0
DRAW_SKELETON = True   # Dibuja el esqueleto completo además de extremidades
SHOW_FPS      = True
# ============================================

# Índices COCO-17 (Ultralytics): 0:nose, 1:leye, 2:reye, 3:lear, 4:rear,
# 5:lshoulder, 6:rshoulder, 7:lelbow, 8:relbow, 9:lwrist, 10:rwrist,
# 11:lhip, 12:rhip, 13:lknee, 14:rknee, 15:lankle, 16:rankle
KPT = {
    "LWrist": 9, "RWrist": 10,
    "LAnkle": 15, "RAnkle": 16,
    "LElbow": 7, "RElbow": 8,
    "LKnee": 13, "RKnee": 14
}

# Pares para esqueleto simple (opcional)
SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # hombros->codos->muñecas
    (11, 12), (5, 11), (6, 12),                   # cadera->hombros
    (11, 13), (13, 15), (12, 14), (14, 16)        # cadera->rodillas->tobillos
]

def open_camera(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def draw_kpt(img, xy, label=None, radius=6, thickness=2):
    x, y = int(xy[0]), int(xy[1])
    cv2.circle(img, (x, y), radius, (0, 255, 255), -1)
    if label:
        cv2.putText(img, label, (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness, cv2.LINE_AA)

def draw_line(img, a, b, thickness=2):
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    cv2.line(img, (ax, ay), (bx, by), (255, 255, 0), thickness)

def label_extremities(frame, kpts_xy):
    """
    Recibe kpts_xy shape (K,2) para una persona.
    Dibuja manos (muñecas) y pies (tobillos). También codos/rodillas como apoyo.
    """
    out = frame
    def safe_get(idx):
        if idx is None or idx >= len(kpts_xy): return None
        pt = kpts_xy[idx]
        if pt is None or np.isnan(pt[0]) or np.isnan(pt[1]): return None
        return pt

    # Manos (muñecas)
    for name in ["LWrist", "RWrist"]:
        pt = safe_get(KPT[name])
        if pt is not None:
            draw_kpt(out, pt, label=name)

    # Pies (tobillos)
    for name in ["LAnkle", "RAnkle"]:
        pt = safe_get(KPT[name])
        if pt is not None:
            draw_kpt(out, pt, label=name)

    # Opcional: codos y rodillas para mejor contexto
    for name in ["LElbow", "RElbow", "LKnee", "RKnee"]:
        pt = safe_get(KPT[name])
        if pt is not None:
            draw_kpt(out, pt, label=name, radius=5, thickness=1)

    return out

def draw_skeleton(img, kpts_xy):
    for (i, j) in SKELETON_PAIRS:
        if i < len(kpts_xy) and j < len(kpts_xy):
            a, b = kpts_xy[i], kpts_xy[j]
            if not (np.isnan(a[0]) or np.isnan(a[1]) or np.isnan(b[0]) or np.isnan(b[1])):
                draw_line(img, a, b)

def main():
    print("[INFO] Cargando modelo de pose...")
    model = YOLO(MODEL_WEIGHTS)  # descarga si no existe
    cap = open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {CAMERA_INDEX}. Cambia el índice o cierra otras apps.")
        return

    print("[INFO] Presiona 'q' para salir. 's' para guardar un frame.")
    t0 = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame no válido.")
            break

        # Infiere pose
        results = model.predict(frame, conf=CONF_THRESH, imgsz=640, verbose=False)

        # Copia para dibujar (así conservas el frame original si quieres)
        vis = frame.copy()

        # Itera personas detectadas
        r = results[0]
        # r.keypoints.xy: (N, K, 2) en píxeles
        kpts = getattr(r, "keypoints", None)
        if kpts is not None and hasattr(kpts, "xy"):
            kpts_xy = kpts.xy  # tensor / ndarray
            if hasattr(kpts_xy, "cpu"):
                kpts_xy = kpts_xy.cpu().numpy()

            # Dibuja esqueleto general y extremidades por persona
            for person_idx in range(kpts_xy.shape[0]):
                person_kpts = kpts_xy[person_idx]  # (K,2)
                if DRAW_SKELETON:
                    draw_skeleton(vis, person_kpts)
                vis = label_extremities(vis, person_kpts)

        # FPS simple
        frame_count += 1
        if frame_count >= 10:
            t1 = time.time()
            fps = frame_count / (t1 - t0)
            t0 = t1
            frame_count = 0

        if SHOW_FPS:
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 220, 50), 2, cv2.LINE_AA)

        cv2.imshow("Pose / Extremidades (YOLO-Pose)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"pose_{int(time.time())}.jpg"
            cv2.imwrite(fname, vis)
            print(f"[INFO] Guardado {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
