import pygame
import random
import math

# Inicializar Pygame
pygame.init()

# Constantes
WIDTH, HEIGHT = 800, 800
CENTER = (WIDTH // 2, HEIGHT // 2)
RADIUS = 350
NUM_BALLS = 10

# Pantalla
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pelotas Evolutivas")
clock = pygame.time.Clock()

# Clase pelota
class Ball:
    def __init__(self):
        self.radius = 10
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0, RADIUS - self.radius)
        self.x = CENTER[0] + r * math.cos(angle)
        self.y = CENTER[1] + r * math.sin(angle)
        speed = 3
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    def update(self):
        self.x += self.vx
        self.y += self.vy

        # Rebote en el borde del círculo
        dx = self.x - CENTER[0]
        dy = self.y - CENTER[1]
        distance = math.hypot(dx, dy)
        if distance + self.radius > RADIUS:
            nx, ny = dx / distance, dy / distance
            dot = self.vx * nx + self.vy * ny
            self.vx -= 2 * dot * nx
            self.vy -= 2 * dot * ny
            overlap = (distance + self.radius) - RADIUS
            self.x -= nx * overlap
            self.y -= ny * overlap

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius))

    def is_colliding(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.hypot(dx, dy)
        return distance < self.radius + other.radius

# Crear pelotas
balls = [Ball() for _ in range(NUM_BALLS)]

# Bucle principal
running = True
while running:
    screen.fill((30, 30, 30))
    pygame.draw.circle(screen, (200, 200, 200), CENTER, RADIUS, 2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Actualizar pelotas
    for ball in balls:
        ball.update()

    # Detectar colisiones
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            a = balls[i]
            b = balls[j]
            if a.is_colliding(b):
                # Elegir aleatoriamente quién se achica
                shrink, grow = (a, b) if random.random() < 0.5 else (b, a)
                if shrink.radius > 3:
                    shrink.radius *= 0.8
                grow.radius *= 1.2
                grow.vx *= 1.1
                grow.vy *= 1.1

    # Eliminar pelotas pequeñas
    balls = [ball for ball in balls if ball.radius >= 3]

    # Dibujar pelotas
    for ball in balls:
        ball.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
