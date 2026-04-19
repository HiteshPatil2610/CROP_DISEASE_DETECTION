"""
DenseNet-121 Classifier Training
Classifies disease from cropped leaf ROI
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json, os
import numpy as np
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────
TRAIN_DIR  = "dataset/processed/train/images"
VAL_DIR    = "dataset/processed/val/images"
MODEL_OUT  = "models/densenet_classifier.h5"
IMG_SIZE   = (224, 224)
BATCH      = 32
EPOCHS     = 30
LR         = 1e-4

with open("models/class_names.json") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

def build_model():
    """DenseNet121 with custom classification head."""
    base = DenseNet121(
        weights    = "imagenet",
        include_top= False,
        input_shape= (*IMG_SIZE, 3)
    )
    # Freeze base layers initially
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    return model, base

def get_generators():
    train_gen = ImageDataGenerator(
        rescale          = 1./255,
        rotation_range   = 20,
        width_shift_range= 0.15,
        height_shift_range=0.15,
        shear_range      = 0.1,
        zoom_range       = 0.2,
        horizontal_flip  = True,
        vertical_flip    = True,
        brightness_range = [0.7, 1.3],
        fill_mode        = "nearest"
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode="categorical"
    )
    val_data = val_gen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE,
        batch_size=BATCH, class_mode="categorical"
    )
    return train_data, val_data

def train():
    print(f"🚀 Training DenseNet-121 Classifier")
    print(f"   Classes   : {NUM_CLASSES}")
    print(f"   Epochs    : {EPOCHS}")
    print(f"   Batch Size: {BATCH}\n")

    model, base = build_model()
    train_data, val_data = get_generators()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(LR),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )

    callbacks = [
        ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, verbose=1)
    ]

    # Phase 1: Train head only
    print("📍 Phase 1: Training classification head...")
    history1 = model.fit(train_data, validation_data=val_data, epochs=15, callbacks=callbacks)

    # Phase 2: Fine-tune entire network
    print("\n📍 Phase 2: Fine-tuning full network...")
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR * 0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    history2 = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

    print(f"\n✅ DenseNet saved to: {MODEL_OUT}")

    # Plot
    plot_history(history2)
    return model

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy"); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/training_history.png", dpi=120)
    print("   Plot saved: models/training_history.png")

if __name__ == "__main__":
    train()