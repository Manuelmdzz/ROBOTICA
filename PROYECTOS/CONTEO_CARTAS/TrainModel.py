# train_cards.py  (Windows/VSCode friendly)
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.multiprocessing as mp

# --- CONFIG ---
DATA_YAML = Path(r"Yolo_Conteo_cartas-3\data.yaml")  # ajusta si tu ruta es distinta
BASE_WEIGHTS = "yolo11s.pt"  # o "yolo11n.pt" para más velocidad

def main():
    # Opcional: fuerza método spawn en Windows ANTES de crear dataloaders
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # ya estaba configurado

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = YOLO(BASE_WEIGHTS)

    # TIP: si tu dataset es pequeño o para evitar problemas, usa workers=0 en Windows
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,          # sube a 100 cuando verifiques que todo corre bien
        imgsz=640,
        batch=16,
        device=device,
        project="runs_cards",
        name="cards_yolo11",
        workers=0,         # <--- clave en Windows para evitar el error de spawn
        deterministic=True # reproducible en Windows
    )

    # Validación
    model = YOLO(Path("runs_cards") / "cards_yolo11" / "weights" / "best.pt")
    model.val(data=str(DATA_YAML), device=device)

    # Predicción rápida en 5 imágenes de validación
    val_imgs = list((DATA_YAML.parent / "valid" / "images").glob("*.jpg"))[:5]
    if val_imgs:
        model.predict(
            source=[str(p) for p in val_imgs],
            device=device,
            save=True,
            project="runs_cards",
            name="pred_samples"
        )
    print("✅ Listo. Revisa la carpeta runs_cards/")

if __name__ == "__main__":
    main()
