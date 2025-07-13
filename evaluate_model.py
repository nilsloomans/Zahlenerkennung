import os
import numpy as np
import pandas as pd
from collections import defaultdict
from app.inference import predict_image  # Modell-Inferenzfunktion

#  Verzeichnis mit Testbildern (manuell vorbereitet oder von der Web-App generiert)
test_dir = "data/test_images"

#  Manuelles Mapping der Personen – muss mit dem Training übereinstimmen
person_map = {"Nils": 0, "Thanadon": 1, "Tim": 2}

#  Ergebnis-Speicher
results = []

# Genauigkeitstracking: korrekt/gesamt je Ziffer bzw. Person
digit_stats = defaultdict(lambda: {"correct": 0, "total": 0})
person_stats = defaultdict(lambda: {"correct": 0, "total": 0})

#  Schleife über alle Testbilder im Verzeichnis
for filename in os.listdir(test_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            #  Extrahiere Ground Truth aus dem Dateinamen (z. B. Tim_4_2.jpg → Tim, 4)
            person_true, digit_true, _ = filename.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").split("_")
            digit_true = int(digit_true)

            img_path = os.path.join(test_dir, filename)

            #  Modellvorhersage auf dem Bild
            digit_pred, person_pred, digit_conf, person_conf = predict_image(img_path)

            #  Vergleich Vorhersage mit Ground Truth
            digit_correct = digit_pred == digit_true
            person_correct = person_pred == person_true

            #  Statistik aktualisieren
            digit_stats[digit_true]["total"] += 1
            person_stats[person_true]["total"] += 1
            if digit_correct:
                digit_stats[digit_true]["correct"] += 1
            if person_correct:
                person_stats[person_true]["correct"] += 1

            #  Ergebniszeile speichern
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

#  Umwandlung der Ergebnisse in DataFrame zur Auswertung & Speicherung
df = pd.DataFrame(results)

#  Gesamtgenauigkeiten berechnen
digit_accuracy = df["Zahl_richtig"].mean()
person_accuracy = df["Person_richtig"].mean()

#  Genauigkeit pro Ziffer anzeigen
print("\n Genauigkeit pro Zahl:")
for digit in sorted(digit_stats):
    stats = digit_stats[digit]
    acc = stats["correct"] / stats["total"] * 100
    print(f"  ➤ Zahl {digit}: {acc:.2f}% korrekt")

#  Genauigkeit pro Person anzeigen
print("\n Genauigkeit pro Person:")
for person in sorted(person_stats):
    stats = person_stats[person]
    acc = stats["correct"] / stats["total"] * 100
    print(f"  ➤ {person}: {acc:.2f}% korrekt")

#  Zusammenfassende Ausgabe
print(f"\n Gesamt Ziffern-Accuracy: {digit_accuracy:.2%}")
print(f" Gesamt Personen-Accuracy: {person_accuracy:.2%}")

#  CSV-Datei mit allen Einzelvorhersagen speichern
df.to_csv("evaluation_results.csv", index=False)
print("\n Ergebnisse gespeichert unter: evaluation_results.csv")