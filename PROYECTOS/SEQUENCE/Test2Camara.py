# cam_detect.py
from pathlib import Path
from ultralytics import YOLO
import torch, cv2, time, os

# === CONFIG ===
BASE_RUNS_DIR = r"C:\UP\PYTHON\ROBOTICS\PROYECTOS\SEQUENCE\runs_sequence_2"  # <-- corregido
RUN_PATTERN  = "Sequence_yolo11*"     # p.ej. Sequence_yolo11_800x600, _640, etc.
IMGSZ        = (800, 608)             # (W, H) acorde a tu entrenamiento
CONF, IOU    = 0.3, 0.45              # <-- baja el conf para no perder detecciones
CAM_INDEX    = 0
WINDOW_NAME  = "YOLO11 Live"
SAVE_DIR     = r"C:\UP\PYTHON\ROBOTICS\PROYECTOS\SEQUENCE\runs_sequence_2\pred_cam"  # <-- corregido

def latest_best_pt(base_dir: str, pattern: str) -> Path:
    base = Path(base_dir)
    cands = [p for p in base.glob(pattern) if (p / "weights" / "best.pt").exists()]
    if not cands:
        raise FileNotFoundError(
            f"No encontré 'best.pt' bajo {base_dir}\\{pattern}. "
            "Verifica que ya entrenaste y existe la carpeta 'weights/best.pt'."
        )
    newest = max(cands, key=lambda p: (p / "weights" / "best.pt").stat().st_mtime)
    return newest / "weights" / "best.pt"

def main():
    ckpt = latest_best_pt(BASE_RUNS_DIR, RUN_PATTERN)
    print(f"Usando checkpoint: {ckpt}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    model = YOLO(str(ckpt))

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMGSZ[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMGSZ[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara. Cambia CAM_INDEX o revisa permisos.")

    os.makedirs(SAVE_DIR, exist_ok=True)
    prev_t = time.perf_counter()
    mirror = True
    recording = False
    writer = None

    print("ESC: salir | S: guardar frame | M: espejo on/off | V: grabar video on/off")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️  No hay frame de la cámara.")
            break

        frame = cv2.flip(cv2.resize(frame, IMGSZ), 1)

        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=device,
            verbose=False
        )

        annotated = results[0].plot()

        now = time.perf_counter()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"conf={CONF:.2f} iou={IOU:.2f} mirror={'on' if mirror else 'off'} rec={'on' if recording else 'off'}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # Grabar si está activo
        if recording:
            if writer is None:
                ts = time.strftime("%Y%m%d-%H%M%S")
                out_path = str(Path(SAVE_DIR) / f"cam_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 30, IMGSZ)
                print(f"Grabando en: {out_path}")
            writer.write(annotated)

        cv2.imshow(WINDOW_NAME, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (ord('s'), ord('S')):
            fname = time.strftime("%Y%m%d-%H%M%S") + ".jpg"
            out_path = str(Path(SAVE_DIR) / fname)
            cv2.imwrite(out_path, annotated)
            print(f"Frame guardado en: {out_path}")
        elif key in (ord('m'), ord('M')):
            mirror = not mirror
        elif key in (ord('v'), ord('V')):
            recording = not recording
            if not recording and writer is not None:
                writer.release()
                writer = None
                print("Grabación detenida.")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Fin de la sesión.")

if __name__ == "__main__":
    main()
