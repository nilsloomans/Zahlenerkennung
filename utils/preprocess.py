import os
import numpy as np
from PIL import Image

def preprocess_image(img_path, output_size=(128, 128)):
    """
    Lädt ein Bild, konvertiert es zu Graustufen, skaliert es und gibt ein normalisiertes Array zurück.
    """
    try:
        img = Image.open(img_path).convert("L")  # Graustufen
        img = img.resize(output_size)
        img_array = np.array(img) / 255.0  # Normalisierung
        return img_array.reshape(*output_size, 1)  # Kanal-Dimension hinzufügen
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {img_path}: {e}")
        return None

def preprocess_all(input_dir, output_dir, output_size=(128, 128)):
    """
    Verarbeitet alle Bilder im Eingabeordner und speichert sie als .npy im Ausgabeordner.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".jpg", ".npy").replace(".png", ".npy"))
            processed = preprocess_image(input_path, output_size)
            if processed is not None:
                np.save(output_path, processed)
                count += 1
    print(f"{count} Bilder erfolgreich verarbeitet und gespeichert.")
