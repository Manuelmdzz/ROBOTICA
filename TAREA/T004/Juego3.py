import arcade
import random
import cv2
import threading
import time
import numpy as np
import sys

# ==============================================================================
# 1. CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

# --- Constantes de la Ventana y el Juego ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
TITLE = "Mouth Match: Asigna la Pieza (CV2 + Arcade 3.5)"

GRID_SIZE = 40
COLS = 10
ROWS = 20
FALL_RATE = 0.7  # Segundos por celda de caída automática

# --- Constantes de Detección de Color (HSV para Rojo) ---
# Rojo tiene dos rangos en el espacio de color HSV
LOWER_RED_1 = np.array([0, 100, 100])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([160, 100, 100])
UPPER_RED_2 = np.array([179, 255, 255])

# Área mínima de contorno para considerarlo una 'acción' (ajústalo si es necesario)
MIN_CONTOUR_AREA = 500  


# ==============================================================================
# 2. CLASE DE VISIÓN POR COMPUTADORA (CV)
# ==============================================================================

class CVRedDetector(threading.Thread):
    """
    Gestiona la captura de cámara y la detección de color en un hilo separado 
    para evitar bloquear el bucle de Arcade.
    """
    def __init__(self, use_camera=True):
        super().__init__()
        self.daemon = True 
        self.stopped = False
        self.new_action = False # Flag que el juego Arcade consulta
        self.use_camera = use_camera

        if not self.use_camera:
            return

        # Inicializa la cámara
        self.cap = cv2.VideoCapture(0)
        # Configuración de baja resolución para mejor rendimiento
        self.CAP_W, self.CAP_H = 320, 240
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAP_H)
        
        if not self.cap.isOpened():
             print("Error: No se puede acceder a la cámara. Ejecutando en modo teclado.")
             self.use_camera = False
             return

        # Definir la Región de Interés (ROI) en el tercio inferior central
        self.ROI_Y_START = int(self.CAP_H * 0.6)
        self.ROI_Y_END = self.CAP_H
        self.ROI_X_START = int(self.CAP_W * 0.25)
        self.ROI_X_END = int(self.CAP_W * 0.75)
        
        self.is_active = False 
        self.active_frames = 0 
        self.ACTIVATION_THRESHOLD = 2 # Frames necesarios para registrar la acción

    def run(self):
        if not self.use_camera: return

        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret: 
                time.sleep(0.05)
                continue
            
            # Recortar la ROI
            roi = frame[self.ROI_Y_START:self.ROI_Y_END, self.ROI_X_START:self.ROI_X_END]
            
            # Convertir a HSV y crear las máscaras de color rojo
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
            mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Filtrar ruido
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Encontrar contornos
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected = False
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                
                if area > MIN_CONTOUR_AREA:
                    detected = True

            # Lógica de detección estable
            if detected:
                self.active_frames += 1
                if self.active_frames >= self.ACTIVATION_THRESHOLD and not self.is_active:
                    self.is_active = True
                    self.new_action = True # ¡Acción detectada!
            else:
                if self.is_active:
                    self.is_active = False 
                self.active_frames = 0
                
            time.sleep(0.01)

    def check_for_action(self):
        """Método llamado por el bucle de Arcade."""
        if self.new_action:
            self.new_action = False
            return True
        return False

    def stop(self):
        """Detiene la captura de cámara y el hilo."""
        self.stopped = True
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()


# ==============================================================================
# 3. CLASES DE JUEGO PYTHON ARCADE
# ==============================================================================

class GameView(arcade.View):
    """
    Vista principal del juego: gestiona la caída, el movimiento y la asignación.
    """
    def __init__(self, use_camera=True):
        super().__init__()
        self.use_camera = use_camera
        
        # Inicializar y empezar el hilo de OpenCV
        self.detector = CVRedDetector(use_camera=use_camera)
        # Si la cámara falla, forzamos el modo teclado
        if not self.detector.use_camera:
             self.use_camera = False
             print("Advertencia: Cámara inactiva. Usando control por teclado.")
        else:
            self.detector.start()
        
        self.score = 0
        self.game_over = False
        self.grid = [[0] * COLS for _ in range(ROWS)] 
        self.current_piece = None
        self.next_piece_color = self.get_random_piece_color()
        self.time_until_fall = FALL_RATE

        arcade.set_background_color(arcade.color.DARK_BLUE_GRAY)
        
    def get_random_piece_color(self):
        """Devuelve un color de pieza aleatorio."""
        # Colores posibles para las piezas
        colors = [arcade.color.RED_BROWN, arcade.color.BLUE, arcade.color.YELLOW, arcade.color.GREEN]
        return random.choice(colors)

    def spawn_piece(self):
        """Crea una nueva pieza en la parte superior."""
        color = self.next_piece_color
        # Usamos SpriteSolidColor ya que no estamos cargando assets
        self.current_piece = arcade.SpriteSolidColor(GRID_SIZE, GRID_SIZE, color)
        self.current_piece.center_x = (COLS // 2) * GRID_SIZE + GRID_SIZE // 2
        self.current_piece.center_y = SCREEN_HEIGHT - GRID_SIZE // 2
        
        self.next_piece_color = self.get_random_piece_color()
        
        # Chequeo de Game Over inmediato
        col = int(self.current_piece.center_x // GRID_SIZE)
        if self.grid[ROWS - 1][col] != 0:
            self.game_over = True

    def on_show_view(self):
        self.spawn_piece()

    def on_draw(self):
        self.clear()
        
        # Dibujar la cuadrícula estática
        for r in range(ROWS):
            for c in range(COLS):
                color = self.grid[r][c]
                if color != 0:
                    x = c * GRID_SIZE + GRID_SIZE // 2
                    y = r * GRID_SIZE + GRID_SIZE // 2
                    arcade.draw_rectangle_filled(x, y, GRID_SIZE, GRID_SIZE, color)
                    arcade.draw_rectangle_outline(x, y, GRID_SIZE, GRID_SIZE, arcade.color.BLACK, 2)
        
        # Dibujar la pieza cayendo
        if self.current_piece:
            self.current_piece.draw()
            arcade.draw_rectangle_outline(
                self.current_piece.center_x, 
                self.current_piece.center_y, 
                GRID_SIZE, GRID_SIZE, arcade.color.BLACK, 2
            )
            
        # Dibujar UI
        arcade.draw_text(f"SCORE: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.WHITE, 18)
        
        ctrl_text = "MUESTRA ALGO ROJO (BOCA) para ASIGNAR!" if self.use_camera else "BARRA ESPACIADORA para ASIGNAR!"
        arcade.draw_text(ctrl_text, SCREEN_WIDTH - 400, SCREEN_HEIGHT - 30, arcade.color.YELLOW, 16)
        
        if self.game_over:
            arcade.draw_rectangle_filled(SCREEN_WIDTH/2, SCREEN_HEIGHT/2, 400, 200, arcade.color.BLACK_OPACITY)
            arcade.draw_text("GAME OVER", SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 30, arcade.color.RED, 40, anchor_x="center")
            arcade.draw_text(f"Puntuación Final: {self.score}\n(Presiona R para reiniciar)", SCREEN_WIDTH/2, SCREEN_HEIGHT/2 - 30, arcade.color.WHITE, 20, anchor_x="center", align="center")


    def update_grid(self):
        """Lógica de limpieza de líneas y puntuación."""
        rows_to_clear = []
        for r in range(ROWS):
            if all(self.grid[r]):
                rows_to_clear.append(r)
                
        lines_cleared = len(rows_to_clear)
        if lines_cleared > 0:
            self.score += lines_cleared * 100 * lines_cleared
            
            new_grid = [row for i, row in enumerate(self.grid) if i not in rows_to_clear]
            
            for _ in range(lines_cleared):
                new_grid.append([0] * COLS)
                
            self.grid = new_grid

    def assign_piece(self):
        """Asigna la pieza y busca la siguiente posición de aterrizaje."""
        if not self.current_piece or self.game_over: return

        # Calcular la columna
        col = int(self.current_piece.center_x // GRID_SIZE)
        
        # Buscar la posición de fila más alta vacía
        final_row = 0
        while final_row < ROWS and self.grid[final_row][col] == 0:
            final_row += 1
        
        final_row -= 1 

        if final_row < 0:
            self.game_over = True
            return

        # Asignar la pieza
        self.grid[final_row][col] = self.current_piece.color
        self.score += 10 
        self.update_grid()

        # Preparar la siguiente pieza
        self.current_piece = None
        self.spawn_piece()

    def on_update(self, delta_time):
        """Llamado cada frame para actualizar la lógica."""
        if self.game_over: return
        
        # --- Lógica de Control CV (Se ejecuta solo si la cámara está activa) ---
        if self.use_camera and self.detector.check_for_action():
            self.assign_piece()

        # --- Lógica de Caída Automática ---
        self.time_until_fall -= delta_time
        if self.time_until_fall <= 0:
            if self.current_piece:
                self.current_piece.center_y -= GRID_SIZE 
                
                # Para simplificar, si toca el fondo, se asigna
                if self.current_piece.bottom < 0:
                    self.assign_piece()
                
            self.time_until_fall = FALL_RATE

    def on_key_press(self, key, modifiers):
        """Manejo de entrada de teclado (movimiento y control alternativo)."""
        if self.game_over:
            if key == arcade.key.R:
                 self.window.show_view(MenuView())
            return
        
        # Teclado: Movimiento Lateral
        if key == arcade.key.LEFT:
            if self.current_piece and self.current_piece.left > GRID_SIZE:
                self.current_piece.center_x -= GRID_SIZE
        elif key == arcade.key.RIGHT:
            if self.current_piece and self.current_piece.right < SCREEN_WIDTH - GRID_SIZE:
                self.current_piece.center_x += GRID_SIZE
        
        # Teclado: Asignación (Modo Alternativo)
        elif not self.use_camera and key == arcade.key.SPACE:
             self.assign_piece()

    def on_hide_view(self):
        """Detiene el detector CV al salir de la vista."""
        if self.detector.is_alive():
            self.detector.stop()


class MenuView(arcade.View):
    """Vista del menú inicial para seleccionar el modo de control."""
    def on_show_view(self):
        arcade.set_background_color(arcade.color.DARK_BLUE)
        print("Mouth Match iniciado. Presiona C o K.")

    def on_draw(self):
        self.clear()
        arcade.draw_text(TITLE, SCREEN_WIDTH/2, SCREEN_HEIGHT * 0.7,
                         arcade.color.WHITE, 50, anchor_x="center")
        
        arcade.draw_text("Control CV: Muestra algo ROJO (boca) a la cámara", SCREEN_WIDTH/2, SCREEN_HEIGHT * 0.5,
                         arcade.color.YELLOW, 20, anchor_x="center")
        arcade.draw_text("Presiona 'C' (Cámara)", SCREEN_WIDTH/2, SCREEN_HEIGHT * 0.45,
                         arcade.color.WHITE, 24, anchor_x="center")

        arcade.draw_text("Control Teclado: Barra Espaciadora para asignar", SCREEN_WIDTH/2, SCREEN_HEIGHT * 0.3,
                         arcade.color.YELLOW_ORANGE, 20, anchor_x="center")
        arcade.draw_text("Presiona 'K' (Teclado)", SCREEN_WIDTH/2, SCREEN_HEIGHT * 0.25,
                         arcade.color.WHITE, 24, anchor_x="center")
        
        arcade.draw_text("El movimiento lateral siempre es con TECLAS IZQ/DER", SCREEN_WIDTH/2, SCREEN_HEIGHT * 0.15,
                         arcade.color.LIGHT_GRAY, 14, anchor_x="center")

    def on_key_press(self, key, modifiers):
        if key == arcade.key.C:
            game_view = GameView(use_camera=True)
            self.window.show_view(game_view)
        elif key == arcade.key.K:
            game_view = GameView(use_camera=False)
            self.window.show_view(game_view)


# ==============================================================================
# 4. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==============================================================================

def main():
    """Función principal para inicializar el juego y la ventana."""
    print("--- DEPENDENCIAS REQUERIDAS ---")
    print("pip install arcade opencv-python numpy")
    print("-------------------------------")

    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, TITLE)
    menu_view = MenuView()
    window.show_view(menu_view)
    arcade.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n--- ERROR FATAL EN EJECUCIÓN ---")
        print(f"Asegúrate de tener instalados: arcade, opencv-python, y numpy.")
        print(f"Detalle del error: {e}", file=sys.stderr)