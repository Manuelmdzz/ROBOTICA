import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [
        result.names[cls.item()] for cls in result.boxes.cls.int()
    ]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    print("xywh:", xywh)
    print("xywhn:", xywhn)
    print("xyxy:", xyxy)
    print("xyxyn:", xyxyn)
    print("names:", names)
    print("confs:", confs)

show = result.plot()  # return annotated image (np.array)

plt.imshow(show[:, :, ::-1])  # convert BGR to RGB
plt.axis("off")
plt.show()
