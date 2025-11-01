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
ANCHO_JUEGO = 600
ALTO_JUEGO_CAJA = 700  # Alto del contenedor de frutas
ALTO_PANEL_CONTROL = 200 # Aumentamos el espacio en la parte superior para la cámara y mensajes

# Alto TOTAL de la ventana
ALTO_JUEGO = ALTO_JUEGO_CAJA + ALTO_PANEL_CONTROL 

# --- Configuración de Cámara para CV2 ---
# --- CAMBIO: Cámara más ancha para alinear con el juego ---
ANCHO_CAMARA = ANCHO_JUEGO # Aumentado para coincidir con el ancho del juego
ALTO_CAMARA = 150 # Aumentado
# Posición de la cámara en el nuevo panel de control superior
POS_CAMARA_X = 0 # Ajustado a 0 para alinear a la izquierda
# --- FIN DEL CAMBIO ---
POS_CAMARA_Y = 10 
CHROMA_KEY_COLOR = (0, 0, 0) # El color negro puro será transparente (RGB)

# --- CAMBIO: Configuración de Frutas con Imágenes ---
# Ahora el tercer elemento es la RUTA de la imagen
FRUTAS_INFO = [
    (20, "Cereza", "img/cereza.png"),
    (30, "Fresa", "img/fresa.png"),
    (45, "Uva", "img/uva.png"),
    (50, "Melocotón", "img/melocoton.png"),
    (60, "Naranja", "img/naranja.png"),
    (70, "Manzana", "img/manzana.png"),
    (85, "Melón", "img/melon.png"),
    (100, "Sandía", "img/sandia.png"), # <-- ¡FRUTA NUEVA AÑADIDA!
    (120, "Calabaza", "img/calabaza.png"), # <-- ¡FRUTA NUEVA AÑADIDA!
]
# --- FIN DEL CAMBIO ---

radios_frutas = [info[0] for info in FRUTAS_INFO]
MAPA_FUSION = {radios_frutas[i]: radios_frutas[i+1] for i in range(len(radios_frutas) - 1)}
if radios_frutas:
    MAPA_FUSION[radios_frutas[-1]] = radios_frutas[-1] 

# --- CAMBIO: Caché global para imágenes cargadas y fondo ---
IMAGENES_FRUTAS = {}
FONDO_JUEGO = None
# --- FIN DEL CAMBIO ---

# --- Clase Fruta (Misma que antes, solo ajustamos la función dibujar a la nueva altura) ---
class Fruta:
    def __init__(self, radio, pos, space):
        self.radio = radio
        info = next((info for info in FRUTAS_INFO if info[0] == radio), (radio, "Desconocida", "img/default.png"))
        self.nombre = info[1]
        
        # --- CAMBIO: Obtener la imagen pre-cargada del caché ---
        if radio in IMAGENES_FRUTAS:
            self.imagen = IMAGENES_FRUTAS[radio]
        else:
            # Fallback por si la imagen no se cargó (aunque no debería pasar)
            self.imagen = IMAGENES_FRUTAS[20] # Default a cereza
        # --- FIN DEL CAMBIO ---
            
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

    # --- CAMBIO: Función de dibujar modificada para usar imágenes y rotación ---
    def dibujar(self, pantalla_pygame):
        p = self.cuerpo.position
        # Coordenadas: Pymunk (0,0 es abajo) a Pygame (0,0 es arriba).
        # Esta sigue siendo la posición del CENTRO
        p_pygame = (int(p.x), int(ALTO_JUEGO - p.y)) 
        
        # 1. Obtener el ángulo de Pymunk (en radianes) y convertir a grados
        # Pymunk usa radianes, Pygame usa grados
        angulo_grados = np.degrees(self.cuerpo.angle)
        
        # 2. Rotar la imagen original (self.imagen).
        # Usamos -angulo_grados porque Pygame rota en sentido anti-horario
        imagen_rotada = pygame.transform.rotate(self.imagen, -angulo_grados)
        
        # 3. Calcular el nuevo rectángulo y centrarlo
        # Al rotar, el tamaño de la imagen cambia. Necesitamos obtener su nuevo
        # rectángulo (rect) y decirle que su centro debe ser nuestra posición de Pymunk.
        rect = imagen_rotada.get_rect(center=p_pygame)
        
        # 4. Dibujar la imagen rotada en la posición top-left del nuevo rect
        pantalla_pygame.blit(imagen_rotada, rect.topleft)
        
        # Opcional: Dibujar nombre (puede verse mal con las imágenes)
        # fuente = pygame.font.Font(None, 20)
        # texto = fuente.render(self.nombre, 1, (0, 0, 0))
        # pantalla_pygame.blit(texto, (p_pygame[0] - texto.get_width()//2, p_pygame[1] - texto.get_height()//2))
    # --- FIN DEL CAMBIO ---

# --- Funciones Auxiliares (crear_paredes_pymunk y fusion_callback son iguales) ---

def crear_paredes_pymunk(space):
    # ... (código sin cambios) ...
    suelo = pymunk.Segment(space.static_body, (0, 0), (ANCHO_JUEGO, 0), 5)
    pared_izquierda = pymunk.Segment(space.static_body, (0, 0), (0, ALTO_JUEGO_CAJA), 5)
    pared_derecha = pymunk.Segment(space.static_body, (ANCHO_JUEGO, 0), (ANCHO_JUEGO, ALTO_JUEGO_CAJA), 5)
    techo_caida = pymunk.Segment(space.static_body, (0, ALTO_JUEGO_CAJA), (ANCHO_JUEGO, ALTO_JUEGO_CAJA), 5)
    techo_caida.collision_type = 99
    for s in [suelo, pared_izquierda, pared_derecha, techo_caida]:
        s.elasticity = 0.5
        s.friction = 1.0
        if s != techo_caida:
             s.collision_type = 0 
        space.add(s)


def fusion_callback(arbiter, space, data):
    # ... (código sin cambios) ...
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
            data['frutas'].append(nueva_fruta) 
    return True

# --- CAMBIO: Nueva función para cargar y escalar imágenes ---
def cargar_recursos():
    global FONDO_JUEGO, IMAGENES_FRUTAS
    
    print("Cargando recursos...")
    try:
        # Cargar y escalar el fondo del área de juego
        fondo_img = pygame.image.load("img/fondo_juego.jpg").convert()
        FONDO_JUEGO = pygame.transform.scale(fondo_img, (ANCHO_JUEGO, ALTO_JUEGO_CAJA))

        # Cargar y escalar todas las imágenes de las frutas
        for radio, nombre, img_path in FRUTAS_INFO:
            try:
                # .convert_alpha() es crucial para imágenes con transparencia (PNG)
                imagen = pygame.image.load(img_path).convert_alpha()
                diametro = int(radio * 2)
                imagen_escalada = pygame.transform.scale(imagen, (diametro, diametro))
                IMAGENES_FRUTAS[radio] = imagen_escalada
                print(f" - Cargada {nombre} ({img_path})")
            except pygame.error as e:
                print(f"ERROR: No se pudo cargar la imagen de fruta: {img_path}")
                print(e)
                # Crear un 'fallback' de círculo de color si la imagen falla
                fallback_surf = pygame.Surface((radio*2, radio*2), pygame.SRCALPHA)
                fallback_surf.fill((0,0,0,0)) # Transparente
                pygame.draw.circle(fallback_surf, (255, 0, 255), (radio, radio), radio) # Círculo magenta
                IMAGENES_FRUTAS[radio] = fallback_surf

    except pygame.error as e:
        print(f"ERROR: No se pudo cargar la imagen de fondo: img/fondo_juego.jpg")
        print(e)
        # Si el fondo falla, simplemente no se dibujará
        FONDO_JUEGO = None
    except FileNotFoundError as e:
        print(f"ERROR: No se encontró un archivo de imagen. Asegúrate de tener la carpeta 'img/' con las imágenes.")
        print(e)
        pygame.quit()
        sys.exit()
# --- FIN DEL CAMBIO ---

# --- Inicialización ---

pygame.init()
pantalla = pygame.display.set_mode((ANCHO_JUEGO, ALTO_JUEGO))
pygame.display.set_caption("Fruit Merge con Hand Control (MediaPipe)")
reloj = pygame.time.Clock()
pygame.font.init()

# --- CAMBIO: Llamar a la función para cargar imágenes ---
cargar_recursos()
# --- FIN DEL CAMBIO ---

# Inicialización de Pymunk
space = pymunk.Space()
space.gravity = GRAVEDAD

# Crear los límites del juego (usando ALTO_JUEGO_CAJA)
crear_paredes_pymunk(space)

# Inicialización de OpenCV
cap = cv2.VideoCapture(0)

# --- INICIALIZACIÓN DE MEDIAPIPE HANDS ---
# ... (código sin cambios) ...
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils


# --- Variables del Juego ---
# ... (código sin cambios) ...
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
    # ... (código sin cambios) ...
    radio_inicial = random.choice(radios_frutas[:3])
    f = Fruta(radio_inicial, (ANCHO_JUEGO / 2, ALTO_JUEGO_CAJA - radio_inicial), space) 
    return f

# --- Estados del juego ---
# ... (código sin cambios) ...
MENU = 0
JUGANDO = 1
FINALIZADO = 2
ESTADO_ACTUAL = MENU

# --- Bucle Principal del Juego ---
running = True
while running:
    # ... (código de eventos sin cambios) ...
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            running = False
        if ESTADO_ACTUAL == JUGANDO and evento.type == pygame.KEYDOWN and evento.key == pygame.K_SPACE:
            tiempo_actual = pygame.time.get_ticks() / 1000.0
            if fruta_cayendo and (tiempo_actual - tiempo_ultimo_lanzamiento) > DELAY_LANZAMIENTO:
                fruta_cayendo.agregar_a_space()
                frutas.append(fruta_cayendo)
                fruta_cayendo = None 
                tiempo_ultimo_lanzamiento = tiempo_actual
                
    # --- Procesamiento de Cámara y Control de Mano (MediaPipe) ---
    # ... (código sin cambios) ...
    ret, frame_cv2 = cap.read()
    if not ret:
        break
    frame_cv2 = cv2.flip(frame_cv2, 1)
    frame_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    x_posicion_fruta = ANCHO_JUEGO / 2 
    mano_abierta_para_soltar = False
    mano_cerrada_para_agarrar = False
    todos_dedos_extendidos = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_cv2, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_punta_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_posicion_fruta = int(landmark_punta_indice.x * ANCHO_JUEGO)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distancia_pulgar_indice = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            UMBRAL_CERRADA = 0.05
            UMBRAL_ABIERTA = 0.15
            if distancia_pulgar_indice < UMBRAL_CERRADA:
                mano_cerrada_para_agarrar = True
            elif distancia_pulgar_indice > UMBRAL_ABIERTA:
                mano_abierta_para_soltar = True
            dedos_extendidos = []
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
            thumb_is_extended = False
            thumb_tip_lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip_lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            if thumb_tip_lm.y < thumb_ip_lm.y:
                thumb_is_extended = True
            if all(dedos_extendidos) and thumb_is_extended:
                todos_dedos_extendidos = True
    if fruta_cayendo:
        radio = fruta_cayendo.radio
        x_posicion_fruta = max(radio, min(ANCHO_JUEGO - radio, x_posicion_fruta))

    # --- Lógica del juego según el estado ---
    # ... (código sin cambios) ...
    if ESTADO_ACTUAL == MENU:
        if todos_dedos_extendidos:
            ESTADO_ACTUAL = JUGANDO
            frutas = []
            for radio in radios_frutas:
                handler = space.add_collision_handler(radio, radio)
                if handler:
                    handler.data["frutas"] = frutas
            fruta_cayendo = None
            tiempo_ultimo_lanzamiento = 0
    elif ESTADO_ACTUAL == JUGANDO:
        if not fruta_cayendo:
            fruta_cayendo = obtener_nueva_fruta_aleatoria()
        if fruta_cayendo:
            if mano_cerrada_para_agarrar:
                fruta_cayendo.cuerpo.position = (x_posicion_fruta, ALTO_JUEGO_CAJA - fruta_cayendo.radio)
                tiempo_ultimo_lanzamiento = pygame.time.get_ticks() / 1000.0 
            else:
                fruta_cayendo.cuerpo.position = (x_posicion_fruta, ALTO_JUEGO_CAJA - fruta_cayendo.radio)
            if mano_abierta_para_soltar:
                tiempo_actual = pygame.time.get_ticks() / 1000.0
                if fruta_cayendo and (tiempo_actual - tiempo_ultimo_lanzamiento) > DELAY_LANZAMIENTO:
                    fruta_cayendo.agregar_a_space()
                    frutas.append(fruta_cayendo)
                    fruta_cayendo = None 
                    tiempo_ultimo_lanzamiento = tiempo_actual
            # --- CAMBIO: La línea de guía ahora se dibuja sobre el fondo ---
            guia_y_pygame_inicio = ALTO_PANEL_CONTROL
            guia_y_pygame_fin = ALTO_JUEGO
            # Color más tenue para que no distraiga
            pygame.draw.line(pantalla, (255, 255, 255, 100), (x_posicion_fruta, guia_y_pygame_inicio), (x_posicion_fruta, guia_y_pygame_fin), 1)
            # --- FIN DEL CAMBIO ---

        space.step(1.0/FPS)

        frutas_a_eliminar_final = []
        frutas_a_añadir_final = []
        for radio_tipo in radios_frutas:
            # --- CAMBIO: Usar .get_collision_handler() es más seguro ---
            handler = space.add_collision_handler(radio_tipo, radio_tipo)
            # --- FIN DEL CAMBIO ---
            if handler:
                frutas_a_añadir_final.extend(handler.data.get("nuevas_frutas_en_step", []))
                frutas_a_eliminar_final.extend(handler.data.get("frutas_a_eliminar_en_step", []))
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
        for fruta in frutas:
            fruta.fusionada = False

    # --- Dibujo en Pygame ---
    pantalla.fill((0, 0, 50)) # Fondo oscuro principal
    
    # --- CAMBIO: Dibujar la imagen de fondo en el área de juego ---
    if FONDO_JUEGO:
        # Dibujamos el fondo en la posición (0, ALTO_PANEL_CONTROL)
        pantalla.blit(FONDO_JUEGO, (0, ALTO_PANEL_CONTROL))
    # --- FIN DEL CAMBIO ---
    
    # Dibujar la línea divisoria
    pygame.draw.line(pantalla, (255, 255, 255), (0, ALTO_PANEL_CONTROL), (ANCHO_JUEGO, ALTO_PANEL_CONTROL), 3)

    # Dibujar las frutas (solo en estado JUGANDO)
    if ESTADO_ACTUAL == JUGANDO:
        for fruta in frutas:
            fruta.dibujar(pantalla)
        if fruta_cayendo:
            fruta_cayendo.dibujar(pantalla)
            
    # --- Integrar la Cámara de CV2 con Transparencia ---
    frame_cv2_resized = cv2.resize(frame_cv2, (ANCHO_CAMARA, ALTO_CAMARA))
    # ... (código sin cambios) ...
    frame_cv2_rgb_pygame = cv2.cvtColor(frame_cv2_resized, cv2.COLOR_BGR2RGB)
    surface_cv2 = pygame.surfarray.make_surface(frame_cv2_rgb_pygame.swapaxes(0, 1))
    surface_cv2.set_colorkey(CHROMA_KEY_COLOR)
    pantalla.blit(surface_cv2, (POS_CAMARA_X, POS_CAMARA_Y))
    
    # Mostrar mensajes en el panel de control
    # ... (código sin cambios) ...
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

