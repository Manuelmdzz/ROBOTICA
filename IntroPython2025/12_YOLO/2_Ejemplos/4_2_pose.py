import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("frame", frame)
    cv2.imshow("frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
