# validate_only.py
from ultralytics import YOLO
from pathlib import Path
import torch

# === RUTAS AJUSTA ESTO A TU PC ===
CKPT = r"C:\UP\PYTHON\ROBOTICS\PROYECTOS\SEQUENCE\runs_cards\Sequence_yolo112\weights\best.pt"
DATA = r"C:\UP\PYTHON\ROBOTICS\PROYECTOS\SEQUENCE\Sequence_Tokens-1\data.yaml"

device = 0 if torch.cuda.is_available() else "cpu"
print("Usando dispositivo:", device)

model = YOLO(CKPT)

metrics = model.val(
    data=DATA,
    imgsz=(800, 600),
    device=device,
    workers=0,     # clave en Windows
    plots=True     # guarda curvas, matriz de confusión, etc.
)

# Métricas principales
print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f} | "
      f"P: {metrics.box.mp:.3f} | R: {metrics.box.mr:.3f}")
print("Resultados/plots en:", metrics.save_dir)
