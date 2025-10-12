# cam_test_cards.py
from pathlib import Path
from ultralytics import YOLO
import cv2, time, torch, os

# === CONFIG ===
RUNS_DIR   = r"C:\UP\PYTHON\ROBOTICS\PROYECTOS\CONTEO_CARTAS\runs_cards"
RUN_GLOB   = "cards_yolo11*"       # ejemplo: cards_yolo11, cards_yolo11_2, etc.
FALLBACK   = "yolo11s.pt"          # por si no se encuentra tu best.pt
IMGSZ      = (800, 600)            # como entrenaste
IOU        = 0.45
START_CONF = 0.10                  # arranca bajo para “ver algo” aunque el modelo sea malo
CAM_INDEX  = 0
WINDOW     = "YOLO11 - Cam Test (q/ESC salir, -/+ conf, m espejo, s guardar)"

def latest_best_pt(base_dir: str, pattern: str) -> Path | None:
    base = Path(base_dir)
    cands = [p for p in base.glob(pattern) if (p / "weights" / "best.pt").exists()]
    if not cands:
        return None
    newest = max(cands, key=lambda p: (p / "weights" / "best.pt").stat().st_mtime)
    return newest / "weights" / "best.pt"

def put_text(img, text, org, scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

def main():
    ckpt = latest_best_pt(RUNS_DIR, RUN_GLOB)
    if ckpt is None:
        print("⚠️  No encontré best.pt en tus runs. Cargo fallback:", FALLBACK)
        model = YOLO(FALLBACK)
        model_name = FALLBACK
    else:
        print("Usando checkpoint:", ckpt)
        model = YOLO(str(ckpt))
        model_name = str(ckpt)

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMGSZ[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMGSZ[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara. Cambia CAM_INDEX o cierra apps que la usen.")

    save_dir = Path(RUNS_DIR) / "pred_cam"
    os.makedirs(save_dir, exist_ok=True)

    conf = START_CONF
    mirror = False
    prev_t = time.perf_counter()
    print("Controles: q/ESC salir | -/+ conf | m espejo | s guardar frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Frame vacío de la cámara.")
            break

        # Resize a 800x600 y espejo opcional (para “modo espejo” visual)
        frame = cv2.resize(frame, IMGSZ)
        view = cv2.flip(frame, 1) if mirror else frame

        # Inferencia (sobre la imagen que ves)
        results = model.predict(
            source=view,          # numpy array
            imgsz=IMGSZ,
            conf=conf,
            iou=IOU,
            device=device,
            agnostic_nms=False,
            verbose=False
        )

        annotated = results[0].plot()

        # Stats: FPS y conteo por clase
        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now

        # Conteo por clase
        cls_counts = {}
        if results and len(results[0].boxes) > 0:
            names = results[0].names  # dict: id -> name
            for c in results[0].boxes.cls.int().tolist():
                cls_counts[names[int(c)]] = cls_counts.get(names[int(c)], 0) + 1

        # Overlay
        put_text(annotated, f"Model: {Path(model_name).name}", (10, 24))
        put_text(annotated, f"FPS: {fps:.1f} | conf: {conf:.2f} | mirror: {mirror}", (10, 52))
        if cls_counts:
            y = 80
            for k, v in sorted(cls_counts.items()):
                put_text(annotated, f"{k}: {v}", (10, y), scale=0.6, thickness=1)
                y += 20
        else:
            put_text(annotated, "Sin detecciones", (10, 80), scale=0.7, thickness=2)

        cv2.imshow(WINDOW, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):  # ESC o q
            break
        elif key in (ord('+'), ord('=')):    # subir conf
            conf = min(0.99, conf + 0.05)
        elif key == ord('-'):                # bajar conf
            conf = max(0.01, conf - 0.05)
        elif key in (ord('m'), ord('M')):    # espejo
            mirror = not mirror
        elif key in (ord('s'), ord('S')):    # guardar frame
            fname = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
            out = str(save_dir / fname)
            cv2.imwrite(out, annotated)
            print("Frame guardado en:", out)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Fin.")

if __name__ == "__main__":
    main()
