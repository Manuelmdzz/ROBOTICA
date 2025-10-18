from ultralytics import YOLO
import cv2
import time

# cap = cv2.VideoCapture("Videos/people.mp4")
cap = cv2.VideoCapture(0)
model = YOLO("yolo11n.pt")

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
for _ in range(5):
    cv2.waitKey(1)
time.sleep(1)
