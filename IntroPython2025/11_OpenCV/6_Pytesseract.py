import cv2
import pytesseract
from PIL import Image

img_cv = cv2.imread("CAPTCHA.png")

# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# we need to convert from BGR to RGB format/mode:
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img_rgb))
# OR
img_rgb = Image.frombytes("RGB", img_cv.shape[:2], img_cv, "raw", "BGR", 0, 0)
print("-" * 30)
print(pytesseract.image_to_string(img_rgb))
