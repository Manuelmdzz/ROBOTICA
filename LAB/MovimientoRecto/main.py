#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from pybricks.tools import StopWatch

ev3 = EV3Brick()
ev3.speaker.beep()

MA = Motor(Port.A)
MB = Motor(Port.B)

# Resetear los encoders a 0
MA.reset_angle(0)
MB.reset_angle(0)

# Iniciar cronómetro
watch = StopWatch()

# Avanzar hasta que se recorra 1 metro
radio_rueda_mm = 28     # cambia según tu llanta (ej. rueda EV3 ≈ 28 mm radio → 56 mm diámetro)
perimetro_mm = 2 * 3.1416 * radio_rueda_mm
distancia_objetivo_mm = 450

grados_necesarios = 360 * distancia_objetivo_mm / perimetro_mm

MA.run(500)
MB.run(500)

while abs(MA.angle()) < grados_necesarios:
    pass

tiempo = watch.time()

MA.stop()
MB.stop()
