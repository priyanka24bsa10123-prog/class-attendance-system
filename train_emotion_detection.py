"""
train_emotion_detection.py
---------------------------
Trains a CNN on the FER-2013 dataset to detect 7 facial emotions.

Dataset: FER-2013 (Facial Expression Recognition 2013)
Download: https://www.kaggle.com/datasets/msambare/fer2013
Place the dataset as:
    data/emotions/train/<emotion_class>/image.jpg  ...
    data/emotions/test/<emotion_class>/image.jpg   ...

Emotion classes (7):
    angry, disgust, fear, happy, neutral, sad, surprise

Usage:
    python train_emotion_detection.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_DIR   = "data/emotions/train"
TEST_DIR    = "data/emotions/test"
MODEL_PATH  = "models/emotion_model.h5"
IMG_SIZE    = (48, 48)
BATCH_SIZE  = 64
EPOCHS      = 50
NUM_CLASSES = 7

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# ────────────────────────────────────────────────────────────────────────────


def build_cnn() -> Sequential:
    """
    Build a compact CNN for 48×48 grayscale emotion images.
    Architecture: 3 Conv blocks → Flatten → Dense → Softmax
    """
    model = Sequential([
        # ── Block 1 ──────────────────────────────
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Block 2 ──────────────────────────────
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Block 3 ──────────────────────────────
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Classifier ───────────────────────────
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, test_gen


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("models/training_history.png")
    print("[✓] Training plot saved → models/training_history.png")


def train():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(TRAIN_DIR):
        print("[ERROR] FER-2013 dataset not found.")
        print("  1. Download from https://www.kaggle.com/datasets/msambare/fer2013")
        print("  2. Place images in:")
        print("       data/emotions/train/<emotion>/")
        print("       data/emotions/test/<emotion>/")
        return

    print("[INFO] Loading dataset …")
    train_gen, test_gen = get_data_generators()

    print(f"[INFO] Train samples : {train_gen.samples}")
    print(f"[INFO] Test  samples : {test_gen.samples}")
    print(f"[INFO] Classes       : {list(train_gen.class_indices.keys())}")

    model = build_cnn()
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]

    print(f"\n[INFO] Training for up to {EPOCHS} epochs …")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluate
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"\n[✓] Test Accuracy : {acc * 100:.2f}%")
    print(f"[✓] Model saved   → {MODEL_PATH}")

    plot_history(history)


if __name__ == "__main__":
    print("=" * 45)
    print("     EMOTION DETECTION MODEL TRAINER")
    print("=" * 45)
    train()
