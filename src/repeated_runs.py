"""Repeated train/test experiments based on the original Master's workflow."""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tensorflow import keras

from .common import load_dataset, build_transformer, compile_model

def main(runs=10):
    X, y, encoder = load_dataset()
    scores = []
    for run in range(runs):
        print(f"Run {run + 1}/{runs}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=run
        )
        model = compile_model(build_transformer(
            X_train.shape[1:], len(encoder.classes_), blocks=4, heads=2, key_dim=2
        ))
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]
        model.fit(
            X_train, y_train, epochs=100, validation_split=0.2,
            batch_size=16, callbacks=callbacks, verbose=0
        )
        predicted = np.argmax(model.predict(X_test, verbose=0), axis=1)
        score = precision_score(
            y_test, predicted, average="weighted", zero_division=0
        )
        scores.append(score)
        print(f"Weighted precision: {score:.4f}")
    print(f"Mean weighted precision: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()
