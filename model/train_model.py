import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_dataset(data_dir):
    X = []
    y_digits = []
    y_persons = []

    person_set = set()

    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            parts = file.replace(".npy", "").split("_")
            if len(parts) != 3:
                print(f"Dateiname übersprungen: {file}")
                continue

            person, digit, _ = parts
            digit = int(digit)

            img = np.load(os.path.join(data_dir, file))
            X.append(img)
            y_digits.append(digit)
            y_persons.append(person)
            person_set.add(person)

    X = np.array(X)
    y_digits = to_categorical(y_digits, 10)

    # Mapping für Personen → Zahlencode
    person_list = sorted(list(person_set))
    person_map = {name: i for i, name in enumerate(person_list)}
    y_persons = [person_map[p] for p in y_persons]
    y_persons = to_categorical(y_persons, len(person_map))

    print(f"Datensätze geladen: {len(X)}")
    print(f"Personen: {person_map}")

    return X, y_digits, y_persons, person_map