"""Shared data-loading and model utilities for Transformer-based HAR."""

from pathlib import Path
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler

ACTIONS = [
    "Swipe_left", "Swipe_right", "Wave", "Clap", "Throw", "Arm_cross",
    "Basketball_shoot", "Draw_X", "Draw_circle_clockwise",
    "Draw_circle_counter_clockwise", "Draw_triangle", "Bowling", "Boxing",
    "Baseball_swing", "Tennis_swing", "Arm_curl", "Tennis_serve", "Push",
    "Knock", "Catch", "Pickup_and_throw", "Jog", "Walk", "Sit_to_stand",
    "Stand_to_sit", "Lunge", "Squat",
]

def load_dataset(data_dir="data/raw/inertial", max_seq_length=200):
    """Load UTD-MHAD inertial .mat files and return X, encoded y, encoder."""
    data_dir = Path(data_dir)
    data, labels = [], []
    for action_idx, action in enumerate(ACTIONS, start=1):
        for subject in range(1, 9):
            for trial in range(1, 5):
                path = data_dir / f"a{action_idx}_s{subject}_t{trial}_inertial.mat"
                if not path.exists():
                    continue
                mat = scipy.io.loadmat(path)
                if "d_iner" not in mat:
                    continue
                seq = np.asarray(mat["d_iner"], dtype=np.float32)
                if seq.shape[0] < max_seq_length:
                    seq = np.pad(
                        seq,
                        ((0, max_seq_length - seq.shape[0]), (0, 0)),
                        mode="constant",
                    )
                else:
                    seq = seq[:max_seq_length, :]
                data.append(seq)
                labels.append(action)
    if not data:
        raise FileNotFoundError(
            "No inertial .mat files found. Download the dataset and place it in "
            f"{data_dir}."
        )

    X = np.asarray(data, dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # Preserve the original project's feature standardization workflow.
    shape = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(shape[0], -1)).reshape(shape)
    return X, y, encoder

def augment_data(X, noise_std=0.01):
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return X + noise

def build_transformer(input_shape, num_classes, blocks=6, heads=4, key_dim=4,
                      dense_units=(256, 128), dropout=(0.5, 0.3)):
    """Build the attention/residual classifier used by the project."""
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    for _ in range(blocks):
        y = keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=key_dim
        )(x, x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.ReLU()(y)
        x = keras.layers.Add()([x, y])

    x = keras.layers.GlobalAveragePooling1D()(x)
    for units, rate in zip(dense_units, dropout):
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def compile_model(model):
    schedule = keras.optimizers.schedules.ExponentialDecay(
        1e-3, decay_steps=10000, decay_rate=0.9, staircase=True
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
