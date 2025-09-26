import cv2
import numpy as np
import random
import time

# ================== Configuración ==================
WIDTH, HEIGHT = 900, 600
PADDLE_W, PADDLE_H = 130, 18
PLAYER_Y = HEIGHT - 40
CPU_Y = 22
BALL_RADIUS = 12
START_SPEED = 7.0
SPEED_UP = 0.02         # aceleración por segundo
MAX_SCORE = 7

FONT = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

# ================== Utilidades ==================
def clamp(v, a, b): return a if v < a else (b if v > b else v)

class SmoothVal:
    def __init__(self, alpha=0.35, init=None):
        self.alpha = alpha
        self.val = init
    def update(self, x):
        if x is None:
            return self.val
        if self.val is None:
            self.val = x
        else:
            self.val = self.alpha * x + (1 - self.alpha) * self.val
        return self.val

# ================== Juego ==================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # quita CAP_DSHOW en Linux/Mac si falla
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # Estado inicial
    player_x = WIDTH // 2 - PADDLE_W // 2
    cpu_x = WIDTH // 2 - PADDLE_W // 2
    ball_x, ball_y = WIDTH // 2, HEIGHT // 2
    angle = random.uniform(-0.6, 0.6)  # leve inclinación inicial
    speed = START_SPEED

    vx = speed * np.cos(angle) * (1 if random.random() < 0.5 else -1)
    vy = speed * np.sin(angle) * (1 if random.random() < 0.5 else 1)
    if abs(vy) < 4:  # asegura componente vertical suficiente
        vy = 4 * (1 if vy >= 0 else -1)

    score_p, score_cpu = 0, 0
    paused = False
    last_t = time.time()
    smooth_face_x = SmoothVal(alpha=0.25, init=player_x + PADDLE_W / 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.flip(frame, 1)

        # Tiempo (para acelerar con el tiempo)
        now = time.time()
        dt = now - last_t
        last_t = now

        # Detección de rostro -> control paleta jugador
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        face_cx = None
        if len(faces) > 0:
            # rostro más grande
            (x, y, w, h) = max(faces, key=lambda b: b[2]*b[3])
            face_cx = x + w / 2
            # dibuja un marco discreto
            cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 200, 255), 2)

        cx_smooth = smooth_face_x.update(face_cx)
        if cx_smooth is not None:
            player_x = int(cx_smooth - PADDLE_W / 2)
        # límites paleta jugador
        player_x = clamp(player_x, 0, WIDTH - PADDLE_W)

        # Lógica de pausa
        if not paused:
            # Acelera la bola ligeramente con el tiempo
            speed += SPEED_UP * dt
            # normaliza velocidad manteniendo dirección
            v = np.hypot(vx, vy)
            if v > 0:
                scale = speed / v
                vx *= scale
                vy *= scale

            # Mover bola
            ball_x += vx
            ball_y += vy

            # Rebote paredes laterales
            if ball_x - BALL_RADIUS < 0:
                ball_x = BALL_RADIUS
                vx = -vx
            elif ball_x + BALL_RADIUS > WIDTH:
                ball_x = WIDTH - BALL_RADIUS
                vx = -vx

            # CPU sigue la bola con suavizado
            target = ball_x - PADDLE_W / 2
            cpu_x += (target - cpu_x) * 0.08  # más alto = más difícil
            cpu_x = clamp(int(cpu_x), 0, WIDTH - PADDLE_W)

            # Colisiones con paletas
            # Jugador
            if (PLAYER_Y - BALL_RADIUS <= ball_y <= PLAYER_Y + PADDLE_H + BALL_RADIUS and
                player_x - BALL_RADIUS <= ball_x <= player_x + PADDLE_W + BALL_RADIUS and
                vy > 0):
                # calcula desvío según donde golpea
                hit = (ball_x - (player_x + PADDLE_W/2)) / (PADDLE_W/2)
                vx = speed * hit * 0.9
                vy = -abs(vy)  # hacia arriba
                ball_y = PLAYER_Y - BALL_RADIUS - 1

            # CPU
            if (CPU_Y - BALL_RADIUS <= ball_y <= CPU_Y + PADDLE_H + BALL_RADIUS and
                cpu_x - BALL_RADIUS <= ball_x <= cpu_x + PADDLE_W + BALL_RADIUS and
                vy < 0):
                hit = (ball_x - (cpu_x + PADDLE_W/2)) / (PADDLE_W/2)
                vx = speed * hit * 0.9
                vy = abs(vy)   # hacia abajo
                ball_y = CPU_Y + PADDLE_H + BALL_RADIUS + 1

            # Punto para alguien
            if ball_y < -BALL_RADIUS:
                # jugador anota
                score_p += 1
                # reset bola
                ball_x, ball_y = WIDTH // 2, HEIGHT // 2
                speed = START_SPEED + 0.6 * (score_p + score_cpu)
                angle = random.uniform(-0.6, 0.6)
                vx = speed * np.cos(angle) * (1 if random.random() < 0.5 else -1)
                vy = abs(speed * np.sin(angle))  # hacia abajo para el jugador
            elif ball_y > HEIGHT + BALL_RADIUS:
                # CPU anota
                score_cpu += 1
                ball_x, ball_y = WIDTH // 2, HEIGHT // 2
                speed = START_SPEED + 0.6 * (score_p + score_cpu)
                angle = random.uniform(-0.6, 0.6)
                vx = speed * np.cos(angle) * (1 if random.random() < 0.5 else -1)
                vy = -abs(speed * np.sin(angle))  # hacia arriba para la CPU

        # Dibujo de elementos
        # Fondo barra marcador
        cv2.rectangle(frame, (0, 0), (WIDTH, 60), (0, 0, 0), -1)
        # Marcador
        cv2.putText(frame, f"Jugador: {score_p}", (20, 40), FONT, 0.9, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"CPU: {score_cpu}", (WIDTH-180, 40), FONT, 0.9, (200,200,200), 2, cv2.LINE_AA)
        cv2.putText(frame, "P: Pausa | ESC: Salir", (WIDTH//2 - 150, 40), FONT, 0.7, (180,180,180), 2, cv2.LINE_AA)

        # Paletas
        cv2.rectangle(frame, (player_x, PLAYER_Y), (player_x + PADDLE_W, PLAYER_Y + PADDLE_H), (80, 200, 255), -1)
        cv2.rectangle(frame, (cpu_x, CPU_Y), (cpu_x + PADDLE_W, CPU_Y + PADDLE_H), (255, 120, 80), -1)

        # Bola
        cv2.circle(frame, (int(ball_x), int(ball_y)), BALL_RADIUS, (255,255,255), -1)

        # Mensaje si no hay detección
        if face_cx is None:
            cv2.putText(frame, "No te veo bien. Iluminacion/centro por favor.",
                        (20, HEIGHT - 20), FONT, 0.7, (0,0,255), 2, cv2.LINE_AA)

        # Fin de juego
        if score_p >= MAX_SCORE or score_cpu >= MAX_SCORE:
            winner = "GANASTE!" if score_p > score_cpu else "PERDISTE :("
            cv2.putText(frame, winner, (WIDTH//2 - 140, HEIGHT//2 - 10),
                        FONT, 1.6, (0, 255, 0) if score_p > score_cpu else (0,0,255), 4, cv2.LINE_AA)
            cv2.putText(frame, "ESC para salir | P para reiniciar",
                        (WIDTH//2 - 230, HEIGHT//2 + 40), FONT, 0.9, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Head-Pong (cv2 only)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (ord('p'), ord('P')):
            # Si terminó la partida y presionas P, reinicia
            if score_p >= MAX_SCORE or score_cpu >= MAX_SCORE:
                score_p, score_cpu = 0, 0
                ball_x, ball_y = WIDTH // 2, HEIGHT // 2
                speed = START_SPEED
                angle = random.uniform(-0.6, 0.6)
                vx = speed * np.cos(angle) * (1 if random.random() < 0.5 else -1)
                vy = speed * np.sin(angle)
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
