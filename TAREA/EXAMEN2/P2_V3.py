import cv2
import pygame
import pymunk
import numpy as np
import sys
import random
import mediapipe as mp  # Importamos MediaPipe

# --- Configuración General ---
FPS = 60

# --- Configuración de Pymunk (Físicas) ---
GRAVEDAD = (0, -981)
ANCHO_JUEGO = 500
ALTO_JUEGO_CAJA = 700  # Alto del contenedor de frutas
ALTO_PANEL_CONTROL = 200 # Aumentamos el espacio en la parte superior para la cámara y mensajes

# Alto TOTAL de la ventana
ALTO_JUEGO = ALTO_JUEGO_CAJA + ALTO_PANEL_CONTROL 

# --- Configuración de Cámara para CV2 ---
ANCHO_CAMARA = ANCHO_JUEGO # Aumentado
ALTO_CAMARA = 180 # Aumentado
# Posición de la cámara en el nuevo panel de control superior
POS_CAMARA_X = ANCHO_JUEGO // 2 - ANCHO_CAMARA // 2 
POS_CAMARA_Y = 0 
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
        
        # Opcional: Dibujar nombre (puede ser lento)
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
    
    # Buscar las frutas en la lista global que el handler tiene
    fruta_a = next((f for f in data['frutas'] if f.forma == forma_a), None)
    fruta_b = next((f for f in data['frutas'] if f.forma == forma_b), None)
    
    if not fruta_a or not fruta_b:
        # Esto puede pasar si una fruta recién creada colisiona antes de ser añadida al 'space'
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
            # Añadir a la lista 'nuevas_frutas_en_step' para que el bucle principal la AÑADA AL 'SPACE'
            data['nuevas_frutas_en_step'].append(nueva_fruta)
            
            # --- FIX: Añadir la nueva fruta a la lista de búsqueda (data['frutas']) inmediatamente ---
            # Esto previene que colisiones subsecuentes en el MISMO frame fallen.
            data['frutas'].append(nueva_fruta) 
            # --- FIN DEL FIX ---
    return True

# --- Inicialización ---

pygame.init()
pantalla = pygame.display.set_mode((ANCHO_JUEGO, ALTO_JUEGO))
pygame.display.set_caption("Fruit Merge con Hand Control (MediaPipe)")
reloj = pygame.time.Clock()
pygame.font.init()

# Inicialización de Pymunk
space = pymunk.Space()
space.gravity = GRAVEDAD

# Crear los límites del juego (usando ALTO_JUEGO_CAJA)
crear_paredes_pymunk(space)

# Inicialización de OpenCV
cap = cv2.VideoCapture(0)

# --- INICIALIZACIÓN DE MEDIAPIPE HANDS ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Solo queremos detectar una mano
    min_detection_confidence=0.7, # Aumentar un poco la confianza
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
# --- FIN DE INICIALIZACIÓN DE MEDIAPIPE ---


# --- Variables del Juego ---
frutas = []
fruta_cayendo = None
tiempo_ultimo_lanzamiento = 0
DELAY_LANZAMIENTO = 0.8 

# Configurar los handlers de colisión
for radio in radios_frutas:
    handler = space.add_collision_handler(radio, radio) 
    # MUY IMPORTANTE: 'handler.data' es un diccionario.
    # Le pasamos la referencia a nuestra lista 'frutas'
    handler.data["frutas"] = frutas 
    handler.data["nuevas_frutas_en_step"] = [] 
    handler.data["frutas_a_eliminar_en_step"] = [] 
    handler.post_solve = fusion_callback 

def obtener_nueva_fruta_aleatoria():
    radio_inicial = random.choice(radios_frutas[:3])
    # La posición inicial Y es dentro de la CAJA de juego (Alto de la caja - radio)
    f = Fruta(radio_inicial, (ANCHO_JUEGO / 2, ALTO_JUEGO_CAJA - radio_inicial), space) 
    return f

# --- Estados del juego ---
MENU = 0
JUGANDO = 1
FINALIZADO = 2 # Podríamos usarlo más adelante para un "Game Over"

ESTADO_ACTUAL = MENU # El juego comienza en el menú

# --- Bucle Principal del Juego ---
running = True
while running:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            running = False
        # El lanzamiento con la barra espaciadora solo funciona si estamos en el estado JUGANDO
        if ESTADO_ACTUAL == JUGANDO and evento.type == pygame.KEYDOWN and evento.key == pygame.K_SPACE:
            tiempo_actual = pygame.time.get_ticks() / 1000.0
            if fruta_cayendo and (tiempo_actual - tiempo_ultimo_lanzamiento) > DELAY_LANZAMIENTO:
                fruta_cayendo.agregar_a_space()
                frutas.append(fruta_cayendo)
                fruta_cayendo = None 
                tiempo_ultimo_lanzamiento = tiempo_actual
                
    # --- Procesamiento de Cámara y Control de Mano (MediaPipe) ---
    ret, frame_cv2 = cap.read()
    if not ret:
        break

    frame_cv2 = cv2.flip(frame_cv2, 1) # Espejo
    
    # Convertir la imagen de BGR a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen para detectar manos
    results = hands.process(frame_rgb)

    # Posición X por defecto (centro)
    x_posicion_fruta = ANCHO_JUEGO / 2 
    mano_abierta_para_soltar = False # Para el gesto de soltar
    mano_cerrada_para_agarrar = False # Para el gesto de agarrar
    todos_dedos_extendidos = False # Para el menú "Empezar Juego"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar el esqueleto de la mano en el frame BGR (para la vista previa)
            mp_draw.draw_landmarks(
                frame_cv2, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # --- Calcular la posición X ---
            # Usaremos la punta del dedo índice (Landmark 8) como control
            landmark_punta_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_posicion_fruta = int(landmark_punta_indice.x * ANCHO_JUEGO)

            # --- Detección de gestos (abrir/cerrar mano) ---
            # Distancia entre la punta del pulgar (4) y la punta del índice (8)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            distancia_pulgar_indice = ((thumb_tip.x - index_tip.x)**2 + \
                                      (thumb_tip.y - index_tip.y)**2)**0.5
            
            # Umbrales para detectar mano cerrada/abierta (ajusta estos valores si es necesario)
            UMBRAL_CERRADA = 0.05 # Pulgar e índice muy cerca
            UMBRAL_ABIERTA = 0.15 # Pulgar e índice separados
            
            if distancia_pulgar_indice < UMBRAL_CERRADA:
                mano_cerrada_para_agarrar = True
            elif distancia_pulgar_indice > UMBRAL_ABIERTA:
                mano_abierta_para_soltar = True

            # --- Detección de "todos los dedos extendidos" para el menú ---
            # Comprobamos que las puntas de los dedos están por encima de la base de sus respectivos dedos.
            
            dedos_extendidos = []
            
            # Índice, Corazón, Anular, Meñique (Landmarks de puntas y articulaciones PIP)
            dedos_indices = [
                (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
            ]
            
            for tip, pip in dedos_indices:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                    dedos_extendidos.append(True)
                else:
                    dedos_extendidos.append(False)

            # Para el pulgar, comprobamos que su punta esté más "arriba" (Y menor) que su articulación IP
            thumb_is_extended = False
            thumb_tip_lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip_lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP] # Nudillo medio del pulgar

            if thumb_tip_lm.y < thumb_ip_lm.y:
                thumb_is_extended = True

            if all(dedos_extendidos) and thumb_is_extended:
                todos_dedos_extendidos = True
    
    # Ajustar la posición X para que la fruta no se salga por los lados
    if fruta_cayendo:
        radio = fruta_cayendo.radio
        x_posicion_fruta = max(radio, min(ANCHO_JUEGO - radio, x_posicion_fruta))


    # --- Lógica del juego según el estado ---
    if ESTADO_ACTUAL == MENU:
        if todos_dedos_extendidos:
            ESTADO_ACTUAL = JUGANDO
            frutas = [] # Limpiar frutas si ya había alguna
            
            # --- FIX: Resetear la lista de frutas en los handlers ---
            # Al reiniciar 'frutas', debemos actualizar la referencia en los handlers
            for radio in radios_frutas:
                handler = space.add_collision_handler(radio, radio)
                if handler:
                    handler.data["frutas"] = frutas
            # --- FIN DEL FIX ---

            fruta_cayendo = None
            tiempo_ultimo_lanzamiento = 0
            
    elif ESTADO_ACTUAL == JUGANDO:
        # --- Gestión de la Fruta que Cae ---
        if not fruta_cayendo:
            fruta_cayendo = obtener_nueva_fruta_aleatoria()
        
        if fruta_cayendo:
            # Si la mano está cerrada, la "agarramos" (evitamos que caiga)
            if mano_cerrada_para_agarrar:
                fruta_cayendo.cuerpo.position = (x_posicion_fruta, ALTO_JUEGO_CAJA - fruta_cayendo.radio)
                # También prevenimos el lanzamiento si está agarrada
                tiempo_ultimo_lanzamiento = pygame.time.get_ticks() / 1000.0 
            else: # Si la mano está abierta o neutra, la fruta "flota" en la posición de control
                fruta_cayendo.cuerpo.position = (x_posicion_fruta, ALTO_JUEGO_CAJA - fruta_cayendo.radio)

            # Si la mano está abierta Y ha pasado el tiempo de delay, lanzamos
            if mano_abierta_para_soltar:
                tiempo_actual = pygame.time.get_ticks() / 1000.0
                if fruta_cayendo and (tiempo_actual - tiempo_ultimo_lanzamiento) > DELAY_LANZAMIENTO:
                    fruta_cayendo.agregar_a_space()
                    frutas.append(fruta_cayendo) # Añadir a la lista global al soltar
                    fruta_cayendo = None 
                    tiempo_ultimo_lanzamiento = tiempo_actual
                    
            # Dibujar línea de guía (usando coordenadas Pygame)
            guia_y_pygame_inicio = ALTO_PANEL_CONTROL # Y = 150
            guia_y_pygame_fin = ALTO_JUEGO # Y = 850
            pygame.draw.line(pantalla, (200, 200, 200), (x_posicion_fruta, guia_y_pygame_inicio), (x_posicion_fruta, guia_y_pygame_fin), 1)

        # --- Actualización de Físicas y Manejo de Fusiones ---
        space.step(1.0/FPS)

        frutas_a_eliminar_final = []
        frutas_a_añadir_final = []

        # Recolectar datos de todos los handlers
        for radio_tipo in radios_frutas:
            handler = space.add_collision_handler(radio_tipo, radio_tipo)
            if handler:
                frutas_a_añadir_final.extend(handler.data.get("nuevas_frutas_en_step", []))
                frutas_a_eliminar_final.extend(handler.data.get("frutas_a_eliminar_en_step", []))
                # Limpiar listas para el siguiente frame
                if "nuevas_frutas_en_step" in handler.data:
                    handler.data["nuevas_frutas_en_step"] = []
                if "frutas_a_eliminar_en_step" in handler.data:
                    handler.data["frutas_a_eliminar_en_step"] = []


        for fruta_eliminada in set(frutas_a_eliminar_final):
            fruta_eliminada.remover_de_space()
            if fruta_eliminada in frutas:
                frutas.remove(fruta_eliminada)

        for nueva_fruta in frutas_a_añadir_final:
            nueva_fruta.agregar_a_space()
            # --- FIX: No añadir a 'frutas' aquí ---
            # Ya se añadió en el callback para evitar 'race conditions'
            # frutas.append(nueva_fruta) # <- Esta línea se elimina
        
        for fruta in frutas:
            fruta.fusionada = False

    # --- Dibujo en Pygame ---
    pantalla.fill((0, 0, 50)) # Fondo oscuro principal
    
    # Dibujar la línea divisoria entre el panel de control y el área de juego
    # --- FIX: Corregido error tipográfico ALTO_PANCHO_CONTROL ---
    pygame.draw.line(pantalla, (255, 255, 255), (0, ALTO_PANEL_CONTROL), (ANCHO_JUEGO, ALTO_PANEL_CONTROL), 3)
    # --- FIN DEL FIX ---

    # Dibujar las frutas (solo en estado JUGANDO)
    if ESTADO_ACTUAL == JUGANDO:
        for fruta in frutas:
            fruta.dibujar(pantalla)
        if fruta_cayendo:
            fruta_cayendo.dibujar(pantalla)
            
    # --- Integrar la Cámara de CV2 con Transparencia ---
    
    # Redimensionar y convertir el frame (usando frame_cv2 que ya tiene los dibujos de la mano)
    frame_cv2_resized = cv2.resize(frame_cv2, (ANCHO_CAMARA, ALTO_CAMARA))
    frame_cv2_rgb_pygame = cv2.cvtColor(frame_cv2_resized, cv2.COLOR_BGR2RGB)
    surface_cv2 = pygame.surfarray.make_surface(frame_cv2_rgb_pygame.swapaxes(0, 1))
    
    # Aplicar Chroma Key (Hacer el color negro puro transparente)
    surface_cv2.set_colorkey(CHROMA_KEY_COLOR)
    
    # Dibujar la superficie en el Panel de Control
    pantalla.blit(surface_cv2, (POS_CAMARA_X, POS_CAMARA_Y))
    
    # Mostrar mensajes en el panel de control
    fuente_panel = pygame.font.Font(None, 24)
    fuente_titulo = pygame.font.Font(None, 40)

    if ESTADO_ACTUAL == MENU:
        texto_titulo = fuente_titulo.render("FRUIT MERGE", 1, (255, 255, 255))
        pantalla.blit(texto_titulo, (ANCHO_JUEGO // 2 - texto_titulo.get_width() // 2, ALTO_PANEL_CONTROL // 2 - 40))
        
        texto_instruccion_menu = fuente_panel.render("Extiende todos los dedos para Empezar", 1, (200, 200, 255))
        pantalla.blit(texto_instruccion_menu, (ANCHO_JUEGO // 2 - texto_instruccion_menu.get_width() // 2, ALTO_PANEL_CONTROL // 2 + 10))
    
    elif ESTADO_ACTUAL == JUGANDO:
        texto_control = fuente_panel.render("Control: Mueve Mano / Suelta: Abre / Agarra: Cierra", 1, (255, 255, 255))
        pantalla.blit(texto_control, (ANCHO_JUEGO // 2 - texto_control.get_width() // 2, ALTO_PANEL_CONTROL - 25))

    pygame.display.flip()
    reloj.tick(FPS)

# --- Finalización ---
hands.close() # Cerrar MediaPipe
cap.release()
pygame.quit()
sys.exit()

