import cv2
import numpy as np
import requests
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Cargar el modelo y el procesador
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Descargar y preparar la imagen
url = "https://http2.mlstatic.com/D_NQ_NP_666603-MLM79842261336_102024-O.webp"
image = Image.open(requests.get(url, stream=True).raw)
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Definir los textos para la detección
texts = [["a photo of a desk", "a photo of a chair", "a photo of a plant", "a photo of a light"]]

# Procesar la imagen
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Obtener las predicciones
target_sizes = torch.Tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

i = 0  # Primera imagen
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Dibujar las detecciones en la imagen
for box, score, label in zip(boxes, scores, labels):
    # Convertir las coordenadas a enteros
    box = [int(i) for i in box.tolist()]
    x1, y1, x2, y2 = box
    
    # Dibujar el rectángulo
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Agregar texto con la clase y confianza
    label_text = f"{text[label]}: {score:.2f}"
    cv2.putText(image_cv, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

# Mostrar la imagen
cv2.imshow('Object Detection', image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()