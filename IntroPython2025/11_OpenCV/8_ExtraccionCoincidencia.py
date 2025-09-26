import cv2

# Leer imágenes
img1 = cv2.imread("./ImagenesCam/img1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./ImagenesCam/img2.jpg", cv2.IMREAD_GRAYSCALE)

# Crear objeto SIFT
sift = cv2.SIFT_create()

print("img1:", img1 is None)
print("img2:", img2 is None)

# Encontrar los puntos clave y descriptores
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Crear objeto BFMatcher y encontrar coincidencias
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Aplicar ratio test de Lowe
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# Dibujar las coincidencias
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imshow("Coincidencias", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
