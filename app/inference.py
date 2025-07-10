import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
from utils.preprocess import preprocess_image

# Modell laden
model = tf.keras.models.load_model("model/cnn_model.h5")

# Mapping für Rückübersetzung
person_map = {"Nils": 0, "Thanadon": 1, "Tim": 2}
reverse_person_map = {v: k for k, v in person_map.items()}

def predict_image(img_path, confidence_threshold=0.7):
    """
    Gibt eine Vorhersage für ein gegebenes Bild zurück (Zahl + Person),
    sowie die jeweiligen Konfidenzwerte (0.0–1.0).
    """
    img = preprocess_image(img_path)
    if img is None:
        return "Fehler beim Einlesen", "Fehler", 0.0, 0.0

    img = np.expand_dims(img, axis=0)  # Batch-Dimension hinzufügen

    digit_probs, person_probs = model.predict(img, verbose=0)

    predicted_digit = int(np.argmax(digit_probs))
    digit_confidence = float(np.max(digit_probs))

    person_confidence = float(np.max(person_probs))
    predicted_person_idx = int(np.argmax(person_probs))

    if person_confidence < confidence_threshold:
        predicted_person = "Unbekannt"
    else:
        predicted_person = reverse_person_map.get(predicted_person_idx, "Unbekannt")

    return predicted_digit, predicted_person, digit_confidence, person_confidence
