import pygame
import pymunk
import sys
import random

# --- Configuración de Pymunk (Físicas) ---
FPS = 60
GRAVEDAD = (0, -981)  # Gravedad hacia abajo (Y negativo en Pymunk)
ANCHO_JUEGO = 500
ALTO_JUEGO_CAJA = 700  # Alto del contenedor de frutas
ALTO_PANEL_CONTROL = 50  # Espacio para mensajes de control
ALTO_JUEGO = ALTO_JUEGO_CAJA + ALTO_PANEL_CONTROL 

# --- Configuración de Control ---
VELOCIDAD_MOVIMIENTO = 15 # Velocidad de movimiento lateral de la fruta

# --- Configuración de Frutas ---
# (Radio, Nombre, Color)
FRUTAS_INFO = [
    (20, "Cereza", (255, 0, 0)),
    (25, "Fresa", (255, 100, 100)),
    (30, "Uva", (128, 0, 128)),
    (35, "Melocotón", (255, 165, 0)),
    (40, "Naranja", (255, 140, 0)),
    (45, "Manzana", (0, 255, 0)),
    (60, "Melón", (0, 200, 50)),
]
radios_frutas = [info[0] for info in FRUTAS_INFO]
MAPA_FUSION = {radios_frutas[i]: radios_frutas[i+1] for i in range(len(radios_frutas) - 1)}
if radios_frutas:
    MAPA_FUSION[radios_frutas[-1]] = radios_frutas[-1] 

# --- Clase Fruta ---
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
        # Conversión de Pymunk a Pygame, ajustando por el panel superior
        p_pygame = (int(p.x), int(ALTO_JUEGO - p.y)) 
        
        pygame.draw.circle(pantalla_pygame, self.color, p_pygame, int(self.radio), 0)
        fuente = pygame.font.Font(None, 20)
        texto = fuente.render(self.nombre, 1, (0, 0, 0))
        pantalla_pygame.blit(texto, (p_pygame[0] - texto.get_width()//2, p_pygame[1] - texto.get_height()//2))

# --- Funciones Auxiliares ---

def crear_paredes_pymunk(space):
    """Crea el suelo y las paredes para delimitar el área de juego."""
    suelo = pymunk.Segment(space.static_body, (0, 0), (ANCHO_JUEGO, 0), 5)
    pared_izquierda = pymunk.Segment(space.static_body, (0, 0), (0, ALTO_JUEGO_CAJA), 5)
    pared_derecha = pymunk.Segment(space.static_body, (ANCHO_JUEGO, 0), (ANCHO_JUEGO, ALTO_JUEGO_CAJA), 5)
    
    # Techo invisible para el límite superior de caída de la fruta (Game Over Line)
    techo_caida = pymunk.Segment(space.static_body, (0, ALTO_JUEGO_CAJA), (ANCHO_JUEGO, ALTO_JUEGO_CAJA), 5)
    techo_caida.collision_type = 99 

    for s in [suelo, pared_izquierda, pared_derecha, techo_caida]:
        s.elasticity = 0.5
        s.friction = 1.0
        if s.collision_type != 99:
             s.collision_type = 0 
        space.add(s)

def fusion_callback(arbiter, space, data):
    """Lógica de fusión (igual que antes)."""
    forma_a, forma_b = arbiter.shapes
    if forma_a.collision_type == 0 or forma_b.collision_type == 0: return True
        
    fruta_a = next((f for f in data['frutas'] if f.forma == forma_a), None)
    fruta_b = next((f for f in data['frutas'] if f.forma == forma_b), None)
    if not fruta_a or not fruta_b: return True

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
pygame.display.set_caption("Fruit Merge (Control por Teclado)")
reloj = pygame.time.Clock()
pygame.font.init()

# Inicialización de Pymunk
space = pymunk.Space()
space.gravity = GRAVEDAD
crear_paredes_pymunk(space)

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
    # La fruta comienza en el centro superior del contenedor
    f = Fruta(radio_inicial, (ANCHO_JUEGO / 2, ALTO_JUEGO_CAJA - radio_inicial), space) 
    return f

# --- Bucle Principal del Juego ---
running = True
while running:
    # --- Manejo de Eventos y Control por Teclado ---
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            running = False
        
        # Lanzamiento con ESPACIO
        if evento.type == pygame.KEYDOWN and evento.key == pygame.K_SPACE:
            tiempo_actual = pygame.time.get_ticks() / 1000.0
            if fruta_cayendo and (tiempo_actual - tiempo_ultimo_lanzamiento) > DELAY_LANZAMIENTO:
                fruta_cayendo.agregar_a_space()
                frutas.append(fruta_cayendo)
                fruta_cayendo = None 
                tiempo_ultimo_lanzamiento = tiempo_actual

    # Control de movimiento continuo (mantiene la tecla presionada)
    keys = pygame.key.get_pressed()
    if fruta_cayendo:
        current_x = fruta_cayendo.cuerpo.position.x
        radio = fruta_cayendo.radio
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            new_x = current_x - VELOCIDAD_MOVIMIENTO
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            new_x = current_x + VELOCIDAD_MOVIMIENTO
        else:
            new_x = current_x

        # Asegurar que la fruta se mantenga dentro de los límites
        new_x = max(radio, min(ANCHO_JUEGO - radio, new_x))
        
        # Actualizar la posición X
        fruta_cayendo.cuerpo.position = (new_x, ALTO_JUEGO_CAJA - radio)

    # --- Gestión de la Fruta que Cae ---
    if not fruta_cayendo:
        fruta_cayendo = obtener_nueva_fruta_aleatoria()
    
    # --- Actualización de Físicas y Manejo de Fusiones ---
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
    
    # Dibujar la línea divisoria (entre panel y caja de juego)
    pygame.draw.line(pantalla, (255, 255, 255), (0, ALTO_PANEL_CONTROL), (ANCHO_JUEGO, ALTO_PANEL_CONTROL), 3)

    # Dibujar las frutas
    for fruta in frutas:
        fruta.dibujar(pantalla)
    if fruta_cayendo:
        fruta_cayendo.dibujar(pantalla)
        
        # Dibujar línea de guía para la fruta que cae
        radio = fruta_cayendo.radio
        guia_y_pygame = ALTO_JUEGO_CAJA - radio*2 + ALTO_PANEL_CONTROL
        pygame.draw.line(pantalla, (255, 255, 255), (0, guia_y_pygame), (ANCHO_JUEGO, guia_y_pygame), 1)

    # Mostrar el mensaje de control en el panel superior
    fuente_panel = pygame.font.Font(None, 30)
    texto_control = fuente_panel.render("Mueve: FLECHAS (< >) | Lanza: ESPACIO", 1, (255, 255, 255))
    pantalla.blit(texto_control, (ANCHO_JUEGO // 2 - texto_control.get_width() // 2, 15))

    pygame.display.flip()
    reloj.tick(FPS)

# --- Finalización ---
pygame.quit()
sys.exit()