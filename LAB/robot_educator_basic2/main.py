# motor_demo.py
from ev3dev2.motor import LargeMotor, OUTPUT_B, SpeedDPS, SpeedPercent
from time import sleep

m = LargeMotor(OUTPUT_B)          # Cambia a OUTPUT_A/C/D si aplica

# 1) Velocidad constante por tiempo
m.on(SpeedPercent(50))            # 50% de la velocidad nominal
sleep(2)
m.off(brake=True)

# 2) Giro por grados (dos vueltas = 720°) a 360 deg/s
m.on_for_degrees(SpeedDPS(360), 720, brake=True, block=True)
