import cv2
import numpy as np
import random
import time

# ------------------ Configuración ------------------
WIDTH, HEIGHT = 800, 600         # Tamaño de la ventana de juego
SPAWN_EVERY = 0.7                # Cada cuántos segundos aparece un objeto
BALL_SPEED_RANGE = (4, 8)        # Velocidad vertical de caída (px/frame)
RADIUS_RANGE = (12, 20)          # Radio de objetos
BOMB_PROB = 0.30                 # Probabilidad de que el objeto sea bomba
MAX_LIVES = 3

# Tipografías
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Detector de rostro (Haar Cascade incluido en OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# ------------------ Utilidades ------------------
class FallingObject:
    def __init__(self, w):
        self.r = random.randint(*RADIUS_RANGE)
        self.x = random.randint(self.r, w - self.r)
        self.y = -self.r
        self.speed = random.randint(*BALL_SPEED_RANGE)
        self.is_bomb = random.random() < BOMB_PROB
        # Color: verde bolita, rojo bomba
        self.color = (0, 255, 0) if not self.is_bomb else (0, 0, 255)

    def update(self):
        self.y += self.speed

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, self.color, -1)
        # Icono simple si es bomba (una X)
        if self.is_bomb:
            t = int(self.r * 0.7)
            cv2.line(frame, (self.x - t, self.y - t), (self.x + t, self.y + t), (255, 255, 255), 2)
            cv2.line(frame, (self.x - t, self.y + t), (self.x + t, self.y - t), (255, 255, 255), 2)

    def out_of_bounds(self, h):
        return self.y - self.r > h

    def collides_with_rect(self, rx, ry, rw, rh):
        # Colisión círculo-rectángulo
        cx, cy, r = self.x, self.y, self.r
        closest_x = np.clip(cx, rx, rx + rw)
        closest_y = np.clip(cy, ry, ry + rh)
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx*dx + dy*dy) <= (r*r)

# Suavizado de la caja de cara (para que no “salte”)
class SmoothBox:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.has = False
        self.x = self.y = self.w = self.h = 0

    def update(self, box):
        if box is None:
            self.has = False
            return None
        (x, y, w, h) = box
        if not self.has:
            self.x, self.y, self.w, self.h = x, y, w, h
            self.has = True
        else:
            self.x = int(self.alpha * x + (1 - self.alpha) * self.x)
            self.y = int(self.alpha * y + (1 - self.alpha) * self.y)
            self.w = int(self.alpha * w + (1 - self.alpha) * self.w)
            self.h = int(self.alpha * h + (1 - self.alpha) * self.h)
        return (self.x, self.y, self.w, self.h)

# ------------------ Juego ------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW para Windows; quítalo en Linux/Mac si da problema.
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    # Forzar tamaño de captura y redimensionar al lienzo del juego
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    objects = []
    last_spawn = time.time()

    score = 0
    lives = MAX_LIVES
    smooth_face = SmoothBox(alpha=0.35)
    game_over = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar a lienzo fijo
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        # Espejo para experiencia más natural
        frame = cv2.flip(frame, 1)

        # Detección de rostro
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Tomar el rostro más grande (más cercano)
        face_box = None
        if len(faces) > 0:
            face_box = max(faces, key=lambda b: b[2] * b[3])
        face_box = smooth_face.update(face_box)

        # Spawning de objetos
        now = time.time()
        if not game_over and (now - last_spawn) >= SPAWN_EVERY:
            objects.append(FallingObject(WIDTH))
            last_spawn = now

        # Actualizar y dibujar objetos
        to_remove = []
        for i, obj in enumerate(objects):
            obj.update()
            obj.draw(frame)

            # Colisión con la cara
            if face_box is not None and obj.collides_with_rect(*face_box):
                if obj.is_bomb:
                    lives -= 1
                else:
                    score += 1
                to_remove.append(i)

            # Fuera de pantalla
            elif obj.out_of_bounds(HEIGHT):
                # Penaliza si era bolita y la dejaste pasar (opcional)
                if not obj.is_bomb:
                    score = max(0, score - 1)
                to_remove.append(i)

        # Eliminar recogidos/salidos
        for idx in reversed(to_remove):
            objects.pop(idx)

        # Dibujar rostro (si detectado)
        if face_box is not None:
            (x, y, w, h) = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 200, 255), 2)
            cv2.putText(frame, "Tu cara", (x, y - 8), FONT, 0.6, (80, 200, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No te veo. Muevete al centro o mejora la luz.",
                        (20, 40), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # UI de marcador
        cv2.rectangle(frame, (0, 0), (WIDTH, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Puntos: {score}", (20, 40), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ESC: Salir", (WIDTH - 180, 40), FONT, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        # Vidas
        for i in range(MAX_LIVES):
            color = (0, 255, 0) if i < lives else (50, 50, 50)
            cv2.circle(frame, (200 + i * 28, 20), 8, color, -1)
            cv2.circle(frame, (200 + i * 28, 20), 8, (255, 255, 255), 1)

        if lives <= 0 and not game_over:
            game_over = True
            objects.clear()

        if game_over:
            cv2.putText(frame, "GAME OVER", (WIDTH // 2 - 140, HEIGHT // 2),
                        FONT, 1.6, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, "Presiona ESC para salir",
                        (WIDTH // 2 - 220, HEIGHT // 2 + 50),
                        FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Cara Catch - cv2 only", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
