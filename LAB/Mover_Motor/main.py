# main.py
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Stop
from pybricks.tools import wait

ev3 = EV3Brick()

# Cambia el puerto según tu conexión: A, B, C o D
m = Motor(Port.B)

ev3.speaker.beep()

# 1) Girar a velocidad constante 2 s (velocidad en grados/segundo)
m.run(400)
wait(2000)
m.stop(Stop.BRAKE)    # frena (BRAKE) o usa Stop.COAST para libre

# 2) Girar un ángulo exacto (720° = 2 vueltas) y mantener posición
m.run_angle(400, 720, then=Stop.HOLD, wait=True)

# 3) Ir a una posición absoluta (por ejemplo, 0°)
m.run_target(500, 0, then=Stop.HOLD, wait=True)

ev3.screen.print("Listo")
