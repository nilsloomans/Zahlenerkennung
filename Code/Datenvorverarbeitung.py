import os
from PIL import Image
import pandas as pd

# Parameter
INPUT_DIR = "raw_images/"
OUTPUT_DIR = "processed_images/"
CSV_PATH = "labels.csv"
IMG_SIZE = (128, 128)

# Stelle sicher, dass Output-Verzeichnis existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Liste für CSV-Daten
data = []

# Durchlaufe alle Bilder
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".png") or filename.endswith(".jpg"):

        # Beispiel: Person1_5_2.png
        name_part = filename.split(".")[0]
        person, digit, instance = name_part.split("_")
        digit = int(digit)

        # Bild laden und verarbeiten
        img_path = os.path.join(INPUT_DIR, filename)
        img = Image.open(img_path).convert("L")  # Graustufen
        img = img.resize(IMG_SIZE)

        # Neuer Name + Pfad
        new_filename = f"{person}_{digit}_{instance}.png"
        new_path = os.path.join(OUTPUT_DIR, new_filename)
        img.save(new_path)

        # Für die CSV
        data.append({
            "filename": new_filename,
            "label": digit,
            "author": person.lower()
        })

# Speichere Labels als CSV
df = pd.DataFrame(data)
df.to_csv(CSV_PATH, index=False)

print("✅ Verarbeitung abgeschlossen.")