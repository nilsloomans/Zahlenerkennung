import os
import numpy as np
import pandas as pd
from app.inference import predict_image

# 🔁 Pfad zu Testbildern
test_dir = "data/test_images"

# 🔧 Person-Mapping (wie im Modelltraining)
person_map = {"Nils": 0, "Thanadon": 1, "Tim": 2}

# 📊 Ergebnisse sammeln
results = []

for filename in os.listdir(test_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            person_true, digit_true, _ = filename.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").split("_")
            digit_true = int(digit_true)

            # Bildpfad
            img_path = os.path.join(test_dir, filename)

            # Vorhersage
            digit_pred, person_pred, digit_conf, person_conf = predict_image(img_path)

            # Bewertung
            digit_correct = digit_pred == digit_true
            person_correct = person_pred == person_true

            results.append({
                "Datei": filename,
                "Zahl_ist": digit_true,
                "Zahl_vorhersage": digit_pred,
                "Zahl_richtig": digit_correct,
                "Person_ist": person_true,
                "Person_vorhersage": person_pred,
                "Person_richtig": person_correct,
                "Digit_Confidence": round(digit_conf, 2),
                "Person_Confidence": round(person_conf, 2)
            })

        except Exception as e:
            print(f"❌ Fehler bei {filename}: {e}")

# 📊 In DataFrame wandeln
df = pd.DataFrame(results)

# Genauigkeiten berechnen
digit_accuracy = df["Zahl_richtig"].mean()
person_accuracy = df["Person_richtig"].mean()

# Ergebnisse anzeigen
print(f"✅ Digit-Accuracy: {digit_accuracy:.2%}")
print(f"✅ Person-Accuracy: {person_accuracy:.2%}")
print()
print(df[["Datei", "Zahl_ist", "Zahl_vorhersage", "Person_ist", "Person_vorhersage", "Zahl_richtig", "Person_richtig"]])

# CSV speichern
df.to_csv("evaluation_results.csv", index=False)
print("\n📄 Ergebnisse gespeichert unter: evaluation_results.csv")