# TrainModel.py  (Windows/VSCode friendly)
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.multiprocessing as mp

# --- CONFIG ---
DATA_YAML = Path(r"sequence_tokens-2\data.yaml")  # ruta corregida (minúsculas)
BASE_WEIGHTS = "yolo11s.pt"                        # o "yolo11n.pt" para más velocidad
OUTPUT_PROJECT = "runs_sequence_2"                 # carpeta de salida solicitada
RUN_NAME = "Sequence_yolo11_800x600"               # nombre del experimento

def _find_sample_images(dataset_root: Path, limit: int = 5):
    """
    Intenta encontrar imágenes de validación en rutas típicas:
    val/images, valid/images, validation/images. Toma JPG/PNG.
    """
    candidates = [
        dataset_root / "val" / "images",
        dataset_root / "valid" / "images",
        dataset_root / "validation" / "images",
    ]
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    imgs = []
    for folder in candidates:
        if folder.exists() and folder.is_dir():
            imgs = [p for p in folder.iterdir() if p.suffix.lower() in exts]
            if imgs:
                break
    return [str(p) for p in imgs[:limit]]

def main():
    # Opcional: fuerza método spawn en Windows ANTES de crear dataloaders
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # ya estaba configurado

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Entrenamiento
    model = YOLO(BASE_WEIGHTS)

    # TIP Windows: workers=0 evita problemas con spawn
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,              # ajusta según necesidad
        imgsz=(800, 600),        # alto x ancho
        batch=16,
        device=device,
        project=OUTPUT_PROJECT,  # carpeta de salida
        name=RUN_NAME,           # subcarpeta del experimento
        workers=0,
        deterministic=True
    )

    # Ruta a best.pt recién generado
    save_dir = Path(results.save_dir)            # p.ej. runs_sequence_2/Sequence_yolo11_800x600
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No se encontró best.pt en {best_path}")

    print(f"✅ Entrenamiento completo. best.pt: {best_path}")

    # Validación con el mejor peso
    best_model = YOLO(str(best_path))
    print("🔎 Corriendo validación...")
    best_model.val(data=str(DATA_YAML), device=device)

    # Predicción rápida en imágenes de validación (si existen)
    print("📸 Buscando imágenes de validación para predicciones de muestra...")
    sample_imgs = _find_sample_images(DATA_YAML.parent, limit=5)
    if sample_imgs:
        print(f"Se encontraron {len(sample_imgs)} imágenes. Guardando predicciones en {OUTPUT_PROJECT}/pred_samples/")
        best_model.predict(
            source=sample_imgs,
            device=device,
            save=True,
            project=OUTPUT_PROJECT,
            name="pred_samples"
        )
    else:
        print("⚠️ No se encontraron imágenes en val/valid/validation. Omitiendo predicciones de muestra.")

    print(f"🎯 Listo. Revisa la carpeta {OUTPUT_PROJECT}/")

if __name__ == "__main__":
    main()
