#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import StopWatch, wait
import math

# Inicializar ladrillo y motores
ev3 = EV3Brick()
MA = Motor(Port.A)
MB = Motor(Port.B)

ev3.speaker.beep()

# Parámetros de la trayectoria
VEL_BASE = 300    # avance constante en °/s
AMP = 200         # amplitud de corrección diferencial
FREQ = 0.5        # frecuencia en Hz
DURACION_MS = 8000  # duración total (8 s)

t0 = StopWatch()

while t0.time() < DURACION_MS:
    t = t0.time() / 1000  # tiempo en segundos

    # Corrección sinusoidal diferencial
    correccion = AMP * math.sin(2 * math.pi * FREQ * t)

    # Motor izquierdo y derecho
    vel_A = VEL_BASE + correccion
    vel_B = VEL_BASE - correccion

    MA.run(vel_A)
    MB.run(vel_B)

    wait(20)

# Detener
MA.stop()
MB.stop()
ev3.speaker.beep()

print("Movimiento sinusoidal adelante-atrás (izq-der) finalizado.")
