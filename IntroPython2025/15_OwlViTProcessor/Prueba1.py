import cv2
import numpy as np
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# --- 1. Configuración del Dispositivo (CUDA o CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- 2. Cargar el modelo y el procesador ---
print("Cargando modelo OwlViT...")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
# Mover el modelo al dispositivo (GPU si está disponible)
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
model.eval() # Poner el modelo en modo de evaluación
print("Modelo cargado.")

# --- 3. Definir los textos para la detección ---
# !!! IMPORTANTE: CAMBIA ESTAS ETIQUETAS POR LO QUE QUIERES DETECTAR !!!
# Puedes agregar o quitar elementos en la lista.
texts = [[
    "a photo of a person", 
    "a photo of a cellphone", 
    "a photo of a cup", 
    "a photo of a bottle",
    "a photo of a computer mouse",
    "a photo of a keyboard"
]]

# --- 4. Inicializar la cámara web ---
cap = cv2.VideoCapture(0) # 0 es usualmente la cámara web por defecto

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print('Iniciando cámara... Presiona "q" para salir.')

# --- 5. Bucle principal de procesamiento de video ---
try:
    while True:
        # Leer un fotograma de la cámara
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el fotograma.")
            break

        # Convertir el fotograma de BGR (OpenCV) a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convertir el array de numpy a imagen PIL
        image_pil = Image.fromarray(frame_rgb)

        # --- 6. Procesar la imagen ---
        # Preparar los inputs para el modelo
        inputs = processor(text=texts, images=image_pil, return_tensors="pt").to(device)

        # --- 7. Realizar la inferencia ---
        with torch.no_grad():
            outputs = model(**inputs)

        # --- 8. Post-procesar las salidas ---
        # Obtener el tamaño del fotograma para escalar las cajas correctamente
        target_sizes = torch.Tensor([[frame.shape[0], frame.shape[1]]]).to(device)
        # Convertir las salidas a bounding boxes
        results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

        i = 0  # Índice para la primera (y única) imagen procesada
        text = texts[i]
        
        # Mover resultados a la CPU para poder usarlos con OpenCV
        boxes = results[i]["boxes"].cpu()
        scores = results[i]["scores"].cpu()
        labels = results[i]["labels"].cpu()

        # --- 9. Dibujar las detecciones en el fotograma (frame) ---
        for box, score, label in zip(boxes, scores, labels):
            # Convertir las coordenadas a enteros
            box = [int(i) for i in box.tolist()]
            x1, y1, x2, y2 = box
            
            # Dibujar el rectángulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Agregar texto con la clase y confianza
            label_text = f"{text[label]}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- 10. Mostrar el resultado ---
        cv2.imshow('Detector OWL-ViT (presiona "q" para salir)', frame)

        # Romper el bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- 11. Liberar recursos ---
    print("Cerrando...")
    cap.release()
    cv2.destroyAllWindows()