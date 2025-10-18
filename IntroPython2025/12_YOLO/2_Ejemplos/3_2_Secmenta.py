import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import cv2

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image


# get original image (BGR) to modify

img = results[0].orig_img.copy()


# Access the results
for result in results:
    # masks: (num_objects, H, W)
    masks = result.masks.data
    boxes = result.boxes
    confs = boxes.conf  # confidences
    classes = boxes.cls  # class ids (COCO: person == 0)
    # print("masks shape:", masks.shape) # torch.Size([6, 640, 480])
    # print("confs:", confs) # tensor([0.8985, 0.8849, 0.8628, 0.8223, 0.4611, 0.4428])
    # print("classes:", classes) # tensor([ 5.,  0.,  0.,  0., 11.,  0.])

    for i in range(len(masks)):
        conf = float(confs[i].item())
        cls = float(classes[i].item())

        if cls == 0 and conf >= 0.8:
            mask = masks[i]
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            # ensure mask is binary (0/1) and resize to match original image size
            mask = mask.astype(np.uint8)
            # resize expects (width, height)
            mask_resized = cv2.resize(
                mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            mask_bool = mask_resized.astype(bool)
            # aplicar máscara a las 3 canales (BGR)
            img[mask_bool] = 0
            img[mask] = 0

# usar la imagen modificada para mostrar
show = img


# show = result.plot()  # return annotated image (np.array)
# muestra la imagen
plt.imshow(show[:, :, ::-1])  # convert BGR to RGB
plt.axis("off")
plt.show()
