import cv2

# Iniciar captura de video
cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Calcular la diferencia absoluta entre dos frames
    diff = cv2.absdiff(frame1, frame2)
    gris = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilatado = cv2.dilate(thresh, None, iterations=3)
    contornos, _ = cv2.findContours(dilatado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar rectángulos alrededor de los contornos
    for contorno in contornos:
        if cv2.contourArea(contorno) < 900:
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
