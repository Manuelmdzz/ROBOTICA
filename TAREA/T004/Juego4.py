import cv2
import pygame
import pymunk
import numpy as np
import sys
import random

# --- Configuración General ---
FPS = 60

# --- Configuración de Pymunk (Físicas) ---
GRAVEDAD = (0, -981)
ANCHO_JUEGO = 500
ALTO_JUEGO_CAJA = 700  # Alto del contenedor de frutas
ALTO_PANEL_CONTROL = 100 # Nuevo espacio en la parte superior para la cámara

# Alto TOTAL de la ventana
ALTO_JUEGO = ALTO_JUEGO_CAJA + ALTO_PANEL_CONTROL 

FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml' 

# --- Configuración de Cámara para CV2 ---
ANCHO_CAMARA = 200 
ALTO_CAMARA = 75 
# Posición de la cámara en el nuevo panel de control superior
POS_CAMARA_X = ANCHO_JUEGO // 2 - ANCHO_CAMARA // 2 
POS_CAMARA_Y = 10 
CHROMA_KEY_COLOR = (0, 0, 0) # El color negro puro será transparente (RGB)

# --- Configuración de Frutas (Igual que antes) ---
FRUTAS_INFO = [
    (20, "Cereza", (255, 0, 0)),
    (30, "Fresa", (255, 100, 100)),
    (45, "Uva", (128, 0, 128)),
    (50, "Melocotón", (255, 165, 0)),
    (60, "Naranja", (255, 140, 0)),
    (70, "Manzana", (0, 255, 0)),
    (85, "Melón", (0, 200, 50)),
]
radios_frutas = [info[0] for info in FRUTAS_INFO]
MAPA_FUSION = {radios_frutas[i]: radios_frutas[i+1] for i in range(len(radios_frutas) - 1)}
if radios_frutas:
    MAPA_FUSION[radios_frutas[-1]] = radios_frutas[-1] 

# --- Clase Fruta (Misma que antes, solo ajustamos la función dibujar a la nueva altura) ---
class Fruta:
    def __init__(self, radio, pos, space):
        self.radio = radio
        info = next((info for info in FRUTAS_INFO if info[0] == radio), (radio, "Desconocida", (255, 255, 255)))
        self.nombre = info[1]
        self.color = info[2]
        self.pos = pos
        self.space = space
        self.fusionada = False 

        self.masa = radio * 0.1
        self.momento = pymunk.moment_for_circle(self.masa, 0, self.radio)
        self.cuerpo = pymunk.Body(self.masa, self.momento)
        self.cuerpo.position = pos
        
        self.forma = pymunk.Circle(self.cuerpo, self.radio)
        self.forma.friction = 0.8
        self.forma.elasticity = 0.2
        self.forma.density = 1.0
        self.forma.collision_type = self.radio

    def agregar_a_space(self):
        if self.cuerpo not in self.space.bodies:
            self.space.add(self.cuerpo, self.forma)
        
    def remover_de_space(self):
        if self.cuerpo in self.space.bodies:
             self.space.remove(self.cuerpo, self.forma)

    def dibujar(self, pantalla_pygame):
        p = self.cuerpo.position
        # Coordenadas: Pymunk (0,0 es abajo) a Pygame (0,0 es arriba).
        # Ajustamos por el nuevo ALTO_JUEGO
        p_pygame = (int(p.x), int(ALTO_JUEGO - p.y)) 
        
        pygame.draw.circle(pantalla_pygame, self.color, p_pygame, int(self.radio), 0)
        fuente = pygame.font.Font(None, 20)
        texto = fuente.render(self.nombre, 1, (0, 0, 0))
        pantalla_pygame.blit(texto, (p_pygame[0] - texto.get_width()//2, p_pygame[1] - texto.get_height()//2))

# --- Funciones Auxiliares (crear_paredes_pymunk y fusion_callback son iguales) ---

def crear_paredes_pymunk(space):
    """Crea el suelo y las paredes para delimitar el área de juego."""
    # El suelo está en y=0, es la base de la CAJA de 700px.
    suelo = pymunk.Segment(space.static_body, (0, 0), (ANCHO_JUEGO, 0), 5)
    
    # Las paredes deben ir hasta el límite de la CAJA, no hasta el límite de la ventana
    pared_izquierda = pymunk.Segment(space.static_body, (0, 0), (0, ALTO_JUEGO_CAJA), 5)
    pared_derecha = pymunk.Segment(space.static_body, (ANCHO_JUEGO, 0), (ANCHO_JUEGO, ALTO_JUEGO_CAJA), 5)
    
    # Se añade un techo invisible para el límite superior de caída de la fruta
    techo_caida = pymunk.Segment(space.static_body, (0, ALTO_JUEGO_CAJA), (ANCHO_JUEGO, ALTO_JUEGO_CAJA), 5)
    techo_caida.collision_type = 99 # Un tipo especial para que el juego termine si una fruta grande lo toca

    for s in [suelo, pared_izquierda, pared_derecha, techo_caida]:
        s.elasticity = 0.5
        s.friction = 1.0
        if s != techo_caida:
             s.collision_type = 0 
        space.add(s)

def fusion_callback(arbiter, space, data):
    forma_a, forma_b = arbiter.shapes
    if forma_a.collision_type == 0 or forma_b.collision_type == 0:
        return True
    fruta_a = next((f for f in data['frutas'] if f.forma == forma_a), None)
    fruta_b = next((f for f in data['frutas'] if f.forma == forma_b), None)
    if not fruta_a or not fruta_b:
        return True
    if fruta_a.radio == fruta_b.radio and not fruta_a.fusionada and not fruta_b.fusionada:
        fruta_a.fusionada = True
        fruta_b.fusionada = True
        data['frutas_a_eliminar_en_step'].append(fruta_a)
        data['frutas_a_eliminar_en_step'].append(fruta_b)
        pos_nueva = (fruta_a.cuerpo.position + fruta_b.cuerpo.position) / 2
        nuevo_radio = MAPA_FUSION.get(fruta_a.radio, fruta_a.radio)
        if nuevo_radio != fruta_a.radio: 
            nueva_fruta = Fruta(nuevo_radio, pos_nueva, space)
            data['nuevas_frutas_en_step'].append(nueva_fruta)
    return True

# --- Inicialización ---

pygame.init()
pantalla = pygame.display.set_mode((ANCHO_JUEGO, ALTO_JUEGO))
pygame.display.set_caption("Fruit Merge con Face Control (Panel Superior)")
reloj = pygame.time.Clock()
pygame.font.init()

# Inicialización de Pymunk
space = pymunk.Space()
space.gravity = GRAVEDAD

# Crear los límites del juego (usando ALTO_JUEGO_CAJA)
crear_paredes_pymunk(space)

# Inicialización de OpenCV y Haar Cascade
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    print(f"ERROR: No se pudo cargar el archivo {FACE_CASCADE_PATH}. ¡Revisa tu carpeta!")
    sys.exit()

# --- Variables del Juego ---
frutas = []
fruta_cayendo = None
tiempo_ultimo_lanzamiento = 0
DELAY_LANZAMIENTO = 0.8 

for radio in radios_frutas:
    handler = space.add_collision_handler(radio, radio) 
    handler.data["frutas"] = frutas 
    handler.data["nuevas_frutas_en_step"] = [] 
    handler.data["frutas_a_eliminar_en_step"] = [] 
    handler.post_solve = fusion_callback 

def obtener_nueva_fruta_aleatoria():
    radio_inicial = random.choice(radios_frutas[:3])
    # La posición inicial Y es dentro de la CAJA de juego (Alto de la caja - radio)
    f = Fruta(radio_inicial, (ANCHO_JUEGO / 2, ALTO_JUEGO_CAJA - radio_inicial), space) 
    return f

# --- Bucle Principal del Juego ---
running = True
while running:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            running = False
        
        if evento.type == pygame.KEYDOWN and evento.key == pygame.K_SPACE:
            tiempo_actual = pygame.time.get_ticks() / 1000.0
            if fruta_cayendo and (tiempo_actual - tiempo_ultimo_lanzamiento) > DELAY_LANZAMIENTO:
                fruta_cayendo.agregar_a_space()
                frutas.append(fruta_cayendo)
                fruta_cayendo = None 
                tiempo_ultimo_lanzamiento = tiempo_actual
                
    # --- Procesamiento de Cámara y Control Facial ---
    ret, frame_cv2 = cap.read()
    if not ret:
        break

    frame_cv2 = cv2.flip(frame_cv2, 1) # Espejo
    
    # 1. Aplicar fondo negro puro (CHROMA KEY) a la imagen de CV2
    # Crear un canvas negro puro
    canvas_negro = np.zeros_like(frame_cv2)
    # Copiar el frame_cv2 a una zona del canvas negro (opcional, si quisieras un margen, pero lo haremos completo)
    frame_con_fondo_negro = frame_cv2 

    gray = cv2.cvtColor(frame_con_fondo_negro, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    
    x_posicion_fruta = ANCHO_JUEGO / 2 

    if len(faces) > 0:
        (x, y, w, h) = faces[0] 
        center_x_cv2 = x + w // 2
        ancho_cv2_original = frame_cv2.shape[1]
        
        # Mapear la posición X del centro de la cara al ancho del JUEGO (500px)
        x_posicion_fruta = int(center_x_cv2 * ANCHO_JUEGO / ancho_cv2_original)
        
        # Dibujar el recuadro de la cara en la vista de la cámara
        cv2.rectangle(frame_con_fondo_negro, (x, y), (x+w, y+h), (0, 255, 0), 2) # Color verde para destacar la detección

        # Ajustar la posición X para que la fruta no se salga por los lados
        if fruta_cayendo:
            radio = fruta_cayendo.radio
            x_posicion_fruta = max(radio, min(ANCHO_JUEGO - radio, x_posicion_fruta))

    # --- Gestión de la Fruta que Cae ---
    if not fruta_cayendo:
        fruta_cayendo = obtener_nueva_fruta_aleatoria()
    
    if fruta_cayendo:
        radio = fruta_cayendo.radio
        # La posición Y de la fruta en espera se actualiza en la parte superior de la CAJA
        fruta_cayendo.cuerpo.position = (x_posicion_fruta, ALTO_JUEGO_CAJA - radio) 
        
        # Dibujar línea de guía (usando coordenadas Pygame)
        guia_y_pygame = ALTO_JUEGO_CAJA - radio*2 + ALTO_PANEL_CONTROL # Ajuste por el nuevo panel
        pygame.draw.line(pantalla, (255, 255, 255), (0, guia_y_pygame), (ANCHO_JUEGO, guia_y_pygame), 1)

    # --- Actualización de Físicas y Manejo de Fusiones (Igual que antes) ---
    space.step(1.0/FPS)

    frutas_a_eliminar_final = []
    frutas_a_añadir_final = []

    for radio_tipo in radios_frutas:
        handler = space.add_collision_handler(radio_tipo, radio_tipo)
        frutas_a_añadir_final.extend(handler.data["nuevas_frutas_en_step"])
        frutas_a_eliminar_final.extend(handler.data["frutas_a_eliminar_en_step"])
        handler.data["nuevas_frutas_en_step"] = []
        handler.data["frutas_a_eliminar_en_step"] = []

    for fruta_eliminada in set(frutas_a_eliminar_final):
        fruta_eliminada.remover_de_space()
        if fruta_eliminada in frutas:
            frutas.remove(fruta_eliminada)

    for nueva_fruta in frutas_a_añadir_final:
        nueva_fruta.agregar_a_space()
        frutas.append(nueva_fruta)
    
    for fruta in frutas:
        fruta.fusionada = False

    # --- Dibujo en Pygame ---
    pantalla.fill((0, 0, 50)) # Fondo oscuro principal
    
    # Dibujar la línea divisoria entre el panel de control y el área de juego
    pygame.draw.line(pantalla, (255, 255, 255), (0, ALTO_PANEL_CONTROL), (ANCHO_JUEGO, ALTO_PANEL_CONTROL), 3)

    # Dibujar las frutas
    for fruta in frutas:
        fruta.dibujar(pantalla)
    if fruta_cayendo:
        fruta_cayendo.dibujar(pantalla)
        
    # --- Integrar la Cámara de CV2 con Transparencia ---
    
    # Redimensionar y convertir el frame (usando frame_con_fondo_negro)
    frame_cv2_resized = cv2.resize(frame_con_fondo_negro, (ANCHO_CAMARA, ALTO_CAMARA))
    frame_cv2_rgb = cv2.cvtColor(frame_cv2_resized, cv2.COLOR_BGR2RGB)
    surface_cv2 = pygame.surfarray.make_surface(frame_cv2_rgb.swapaxes(0, 1))
    
    # 2. Aplicar Chroma Key (Hacer el color negro puro transparente)
    surface_cv2.set_colorkey(CHROMA_KEY_COLOR)
    
    # 3. Dibujar la superficie en el Panel de Control
    pantalla.blit(surface_cv2, (POS_CAMARA_X, POS_CAMARA_Y))
    
    # Mostrar el mensaje de control en el panel
    fuente_panel = pygame.font.Font(None, 24)
    texto_control = fuente_panel.render("Control: Mueve la Cara / Lanza: ESPACIO", 1, (255, 255, 255))
    pantalla.blit(texto_control, (ANCHO_JUEGO // 2 - texto_control.get_width() // 2, ALTO_PANEL_CONTROL - 25))

    pygame.display.flip()
    reloj.tick(FPS)

# --- Finalización ---
cap.release()
pygame.quit()
sys.exit()