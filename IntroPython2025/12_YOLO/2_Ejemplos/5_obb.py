import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/boats.jpg")  # predict on an image

# Access the results
for result in results:
    xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
    xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
    names = [
        result.names[cls.item()] for cls in result.obb.cls.int()
    ]  # class name of each box
    confs = result.obb.conf  # confidence score of each box


show = result.plot()  # return annotated image (np.array)
# muestra la imagen
plt.imshow(show[:, :, ::-1])
plt.axis("off")
plt.show()
