import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import tensorflow as tf

# --- Projektinterne Hilfsfunktion ---
from utils.preprocess import preprocess_image  # Importiere Bildvorverarbeitung (Resize, Normalisierung, Graustufen)

# --- Modell laden ---
# CNN-Modell mit zwei Ausgängen (digit + person), trainiert und gespeichert als .h5
model = tf.keras.models.load_model("model/cnn_model.h5")

# --- Mapping für Personenerkennung ---
# Manuelle Zuordnung: Name → Index
person_map = {"Nils": 0, "Thanadon": 1, "Tim": 2}

# Umkehrung für spätere Rückübersetzung: Index → Name
reverse_person_map = {v: k for k, v in person_map.items()}


def predict_image(img_path, confidence_threshold=0.7):
    """
    Führt eine Vorhersage auf einem Einzelbild durch.
    
    Parameter:
        img_path (str): Pfad zum Bild (z. B. "data/test_images/Tim_4_2.jpg")
        confidence_threshold (float): Schwelle zur Personenidentifikation.
                                      Liegt die Konfidenz darunter → "Unbekannt"

    Rückgabe:
        predicted_digit (int): Erkannte Ziffer (0–9)
        predicted_person (str): Erkannte Person oder "Unbekannt"
        digit_confidence (float): Wahrscheinlichkeit für Ziffer
        person_confidence (float): Wahrscheinlichkeit für Person
    """
    
    img = preprocess_image(img_path)  # Vorverarbeitung: 128x128, Graustufen, Normalisierung
    
    if img is None:
        # Fehlerbehandlung bei fehlerhaftem Bild
        return "Fehler beim Einlesen", "Fehler", 0.0, 0.0

    # Modell erwartet Eingabe im Batch-Format → zusätzliche Dimension hinzufügen
    img = np.expand_dims(img, axis=0)

    # Vorhersage mit Multi-Output-Modell: digit_probs & person_probs (jeweils Softmax-Ausgabe)
    digit_probs, person_probs = model.predict(img, verbose=0)

    # Ziffernvorhersage: Index mit höchster Wahrscheinlichkeit
    predicted_digit = int(np.argmax(digit_probs))
    digit_confidence = float(np.max(digit_probs))  # Höchste Wahrscheinlichkeit für die Ziffer

    # Personen-Vorhersage: Index mit höchster Wahrscheinlichkeit
    person_confidence = float(np.max(person_probs))
    predicted_person_idx = int(np.argmax(person_probs))

    # Nur wenn die Konfidenz für die Person oberhalb der Schwelle liegt, wird ein Name zurückgegeben
    if person_confidence < confidence_threshold:
        predicted_person = "Unbekannt"
    else:
        predicted_person = reverse_person_map.get(predicted_person_idx, "Unbekannt")  # Fallback, falls Mapping fehlschlägt

    # Rückgabe beider Vorhersagen + Konfidenzen
    return predicted_digit, predicted_person, digit_confidence, person_confidence