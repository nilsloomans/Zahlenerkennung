import os
import numpy as np
from PIL import Image

# Bildvorverarbeitung: Größe anpassen, Graustufen konvertieren, normalisieren, als Array zurückgeben
def preprocess_image(img_path, output_size=(64, 64)):
    """
    Lädt ein Bild, konvertiert es zu Graustufen, skaliert es und gibt ein normalisiertes Array zurück.
    """
    try:
        img = Image.open(img_path).convert("L")           # Bild in Graustufen konvertieren
        img = img.resize(output_size)                     # Bildgröße anpassen
        img_array = np.array(img) / 255.0                 # Wertebereich auf [0,1] normalisieren
        return img_array.reshape(*output_size, 1)         # Kanal-Dimension hinzufügen (1-Kanal-Bild)
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {img_path}: {e}")
        return None
