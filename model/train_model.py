import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
    y_digits = tf.keras.utils.to_categorical(y_digits, 10)

    person_list = sorted(list(person_set))
    person_map = {name: i for i, name in enumerate(person_list)}
    y_persons = [person_map[p] for p in y_persons]
    y_persons = tf.keras.utils.to_categorical(y_persons, len(person_map))

    print(f"Datensätze geladen: {len(X)}")
    print(f"Personen: {person_map}")

    return X, y_digits, y_persons, person_map

def build_model(input_shape=(128, 128, 1), num_persons=3):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    output_digit = tf.keras.layers.Dense(10, activation="softmax", name="digit")(x)
    output_person = tf.keras.layers.Dense(num_persons, activation="softmax", name="person")(x)

    model = tf.keras.Model(inputs=inputs, outputs=[output_digit, output_person])

    model.compile(
        optimizer="adam",
        loss={"digit": "categorical_crossentropy", "person": "categorical_crossentropy"},
        metrics={"digit": "accuracy", "person": "accuracy"}
    )

    return model

def main():
    X, y_digits, y_persons, person_map = load_dataset("data/processed")
    
    X_train, X_test, y_d_train, y_d_test, y_p_train, y_p_test = train_test_split(
        X, y_digits, y_persons, test_size=0.2, random_state=42
    )

    model = build_model(input_shape=X.shape[1:], num_persons=y_persons.shape[1])

    model.fit(
        X_train,
        {"digit": y_d_train, "person": y_p_train},
        validation_data=(X_test, {"digit": y_d_test, "person": y_p_test}),
        epochs=100,
        batch_size=16
    )

    os.makedirs("model", exist_ok=True)
    model.save("model/cnn_model.h5")
    print("✅ Modell gespeichert unter: model/cnn_model.h5")

if __name__ == "__main__":
    main()
