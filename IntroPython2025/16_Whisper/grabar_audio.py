import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import time

# --- Configuración de la grabación ---
DURACION_SEGUNDOS = 5      # ¿Cuántos segundos quieres grabar?
TASA_MUESTREO = 44100      # Tasa de muestreo (muestras por segundo)
NOMBRE_ARCHIVO_MP3 = "grabacion.mp3"
CANALES = 1                # 1 para mono, 2 para estéreo

try:
    print(f"¡Prepárate! Grabando {DURACION_SEGUNDOS} segundos de audio...")

    # 1. Grabar el audio usando sounddevice
    # 'dtype='int16'' es un formato común que pydub entiende bien
    grabacion = sd.rec(
        int(DURACION_SEGUNDOS * TASA_MUESTREO),
        samplerate=TASA_MUESTREO,
        channels=CANALES,
        dtype='int16' # Usamos int16 para compatibilidad con pydub
    )

    # Espera a que la grabación termine
    sd.wait()

    print("Grabación finalizada. Procesando...")

    # 2. Convertir el array de NumPy (de sounddevice) a un AudioSegment (de pydub)
    
    # 'tobytes()' convierte los datos del array a bytes crudos
    # 'sample_width=2' porque 'int16' usa 2 bytes por muestra
    # 'frame_rate' es la tasa de muestreo
    # 'channels' debe coincidir con la grabación
    audio_segment = AudioSegment(
        data=grabacion.tobytes(),
        sample_width=grabacion.dtype.itemsize, # Debe ser 2 (por int16)
        frame_rate=TASA_MUESTREO,
        channels=CANALES
    )

    # 3. Exportar el audio como MP3 usando pydub
    print(f"Guardando archivo como '{NOMBRE_ARCHIVO_MP3}'...")
    audio_segment.export(NOMBRE_ARCHIVO_MP3, format="mp3", bitrate="192k")

    print(f"¡Éxito! Audio guardado en '{NOMBRE_ARCHIVO_MP3}'")

except Exception as e:
    print(f"Ha ocurrido un error:")
    print(e)
    print("\n¿Instalaste 'sounddevice', 'numpy' y 'pydub'?")
    print("¿Está FFmpeg instalado y accesible en el PATH de tu sistema?")