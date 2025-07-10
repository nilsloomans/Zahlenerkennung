import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataset(data_dir):
    X, y_digits, y_persons = [], [], []
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
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output_digit = tf.keras.layers.Dense(10, activation="softmax", name="digit")(x)
    output_person = tf.keras.layers.Dense(num_persons, activation="softmax", name="person")(x)

    model = tf.keras.Model(inputs=inputs, outputs=[output_digit, output_person])
    model.compile(
        optimizer="adam",
        loss={"digit": "categorical_crossentropy", "person": "categorical_crossentropy"},
        metrics={"digit": "accuracy", "person": "accuracy"}
    )
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["digit_accuracy"], label="Digit Accuracy (Train)")
    plt.plot(history.history["val_digit_accuracy"], label="Digit Accuracy (Val)")
    plt.plot(history.history["person_accuracy"], label="Person Accuracy (Train)")
    plt.plot(history.history["val_person_accuracy"], label="Person Accuracy (Val)")
    plt.legend()
    plt.title("Accuracy")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["digit_loss"], label="Digit Loss (Train)")
    plt.plot(history.history["val_digit_loss"], label="Digit Loss (Val)")
    plt.plot(history.history["person_loss"], label="Person Loss (Train)")
    plt.plot(history.history["val_person_loss"], label="Person Loss (Val)")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    os.makedirs("model", exist_ok=True)
    plt.savefig("model/training_plot.png")
    plt.close()
    print("📈 Trainingsverlauf gespeichert unter: model/training_plot.png")

def apply_augmentation(X, y_digits, y_persons, augment_factor=2):
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    X_aug, y_d_aug, y_p_aug = [], [], []

    for i in range(len(X)):
        x_sample = X[i].reshape(1, *X[i].shape)
        d_label = y_digits[i]
        p_label = y_persons[i]

        for _ in range(augment_factor):
            aug_img = next(datagen.flow(x_sample, batch_size=1))[0]
            X_aug.append(aug_img)
            y_d_aug.append(d_label)
            y_p_aug.append(p_label)

    X_aug = np.array(X_aug)
    y_d_aug = np.array(y_d_aug)
    y_p_aug = np.array(y_p_aug)

    print(f"🔄 {len(X)} → {len(X_aug)} augmentierte Bilder erzeugt.")
    return np.concatenate([X, X_aug]), np.concatenate([y_digits, y_d_aug]), np.concatenate([y_persons, y_p_aug])

def main():
    X, y_digits, y_persons, person_map = load_dataset("data/processed")

    X_train, X_test, y_d_train, y_d_test, y_p_train, y_p_test = train_test_split(
        X, y_digits, y_persons, test_size=0.2, random_state=42
    )

    X_train_aug, y_d_aug, y_p_aug = apply_augmentation(X_train, y_d_train, y_p_train, augment_factor=2)

    model = build_model(input_shape=X.shape[1:], num_persons=y_persons.shape[1])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_aug,
        {"digit": y_d_aug, "person": y_p_aug},
        validation_data=(X_test, {"digit": y_d_test, "person": y_p_test}),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    os.makedirs("model", exist_ok=True)
    model.save("model/cnn_model.h5")
    print("✅ Modell gespeichert unter: model/cnn_model.h5")

    plot_history(history)

if __name__ == "__main__":
    main()
