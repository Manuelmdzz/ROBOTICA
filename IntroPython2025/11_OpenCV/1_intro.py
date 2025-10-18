import cv2

img = cv2.imread("img.jpg", 1)
img = cv2.resize(img, (400, 400), fx=0.5, fy=0.5)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("image", img)

k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
elif k == ord("s"):
    cv2.imwrite("images/lena_copy.png", img)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()
    print("Invalid key pressed")
