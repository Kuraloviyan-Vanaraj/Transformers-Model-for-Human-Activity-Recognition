"""Five-fold cross-validation experiment."""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score
from tensorflow import keras

from .common import load_dataset, augment_data, build_transformer, compile_model

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    X, y, encoder = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42
    )
    X_aug = augment_data(X_train)
    X_train = np.concatenate([X_train, X_aug])
    y_train = np.concatenate([y_train, y_train])

    fold_scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
        print(f"Fold {fold}/5")
        model = compile_model(build_transformer(
            X_train.shape[1:], len(encoder.classes_),
            blocks=6, heads=4, key_dim=4, dense_units=(512, 256)
        ))
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]
        model.fit(
            X_train[train_idx], y_train[train_idx],
            epochs=100,
            validation_data=(X_train[val_idx], y_train[val_idx]),
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )
        _, accuracy = model.evaluate(X_train[val_idx], y_train[val_idx], verbose=0)
        fold_scores.append(accuracy)
        print(f"Validation accuracy: {accuracy:.4f}")

    print(f"Mean 5-fold accuracy: {np.mean(fold_scores):.4f}")

    predicted = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("Held-out weighted precision:",
          precision_score(y_test, predicted, average="weighted", zero_division=0))

if __name__ == "__main__":
    main()
