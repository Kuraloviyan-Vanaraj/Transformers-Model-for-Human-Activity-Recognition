"""Main train/test experiment.

Run from the repository root:
    python -m src.train
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
from tensorflow import keras

from .common import load_dataset, augment_data, build_transformer, compile_model

np.random.seed(42)
tf.random.set_seed(42)

def main():
    X, y, encoder = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_aug = augment_data(X_train)
    X_train = np.concatenate([X_train, X_aug])
    y_train = np.concatenate([y_train, y_train])

    model = compile_model(build_transformer(
        X_train.shape[1:], len(encoder.classes_)
    ))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "results/best_model.keras", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    model.fit(
        X_train, y_train, epochs=100, validation_split=0.2,
        batch_size=16, callbacks=callbacks
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    predicted = np.argmax(model.predict(X_test), axis=1)
    weighted_precision = precision_score(
        y_test, predicted, average="weighted", zero_division=0
    )

    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted precision: {weighted_precision:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predicted))

if __name__ == "__main__":
    main()
