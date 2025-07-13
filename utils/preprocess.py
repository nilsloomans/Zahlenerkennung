import os
import numpy as np
from PIL import Image

def preprocess_image(img_path, output_size=(128, 128)):
    """
    Lädt ein Bild vom gegebenen Pfad, verarbeitet es für das CNN-Modell:
    - Konvertierung zu Graustufen (1 Kanal)
    - Resize auf Standardgröße (default: 128x128)
    - Normalisierung der Pixelwerte (0–1)
    - Rückgabe als Array mit Form (128, 128, 1)

    Rückgabe:
        - NumPy-Array oder None bei Fehler
    """
    try:
        img = Image.open(img_path).convert("L")  # "L" = Luminance → Graustufenbild
        img = img.resize(output_size)            # Einheitliche Eingabegröße
        img_array = np.array(img) / 255.0         # Werte in [0,1] normalisieren
        return img_array.reshape(*output_size, 1) # Kanal-Dimension anhängen → (128, 128, 1)
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {img_path}: {e}")
        return None

def preprocess_all(input_dir, output_dir, output_size=(128, 128)):
    """
    Läuft über alle Bilddateien im Eingabeordner,
    verarbeitet sie mit `preprocess_image` und speichert das Ergebnis
    als .npy-Datei im Zielordner.

    Unterstützt .jpg, .jpeg, .png
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.rsplit(".", 1)[0] + ".npy")
            processed = preprocess_image(input_path, output_size)
            if processed is not None:
                np.save(output_path, processed)  # Bild als NumPy-Array speichern
                count += 1
    print(f"{count} Bilder erfolgreich verarbeitet und gespeichert.")

#  Wenn Skript direkt ausgeführt wird, starte Preprocessing aller Rohbilder
if __name__ == "__main__":
    preprocess_all("data/raw", "data/processed")