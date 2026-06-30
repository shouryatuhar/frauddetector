"""
Train a MobileNetV3-Small classifier to detect screen recaptures.

Usage:
    python train.py

Dataset structure:
    dataset/
        real/
        screen/

Saves:
    model/screen_detector.keras
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sklearn.model_selection import train_test_split

# ----------------------------
# Configuration
# ----------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

DATA_DIR = "dataset"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# Load image paths
# ----------------------------


def load_image_paths_and_labels():
    real_dir = os.path.join(DATA_DIR, "real")
    screen_dir = os.path.join(DATA_DIR, "screen")

    real_paths = [
        os.path.join(real_dir, f)
        for f in os.listdir(real_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    screen_paths = [
        os.path.join(screen_dir, f)
        for f in os.listdir(screen_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    paths = real_paths + screen_paths
    labels = [0] * len(real_paths) + [1] * len(screen_paths)

    print(f"\nFound {len(real_paths)} REAL images")
    print(f"Found {len(screen_paths)} SCREEN images\n")

    return paths, labels


# ----------------------------
# Image loading
# ----------------------------


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)
    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(len(paths), seed=SEED)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ----------------------------
# Build model
# ----------------------------


def build_model():
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ])

    base_model = MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = augmentation(inputs)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.30)(x)

    x = layers.Dense(64, activation="relu")(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base_model


# ----------------------------
# Main
# ----------------------------


def main():

    paths, labels = load_image_paths_and_labels()

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    print(f"Training images : {len(train_paths)}")
    print(f"Validation images : {len(val_paths)}")

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels)

    model, base_model = build_model()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "screen_detector.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # ----------------------------
    # Stage 1
    # ----------------------------

    print("\n========== Stage 1 ==========")
    print("Training classifier head...\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[checkpoint, early_stop],
    )

    # ----------------------------
    # Stage 2
    # ----------------------------

    print("\n========== Stage 2 ==========")
    print("Fine-tuning backbone...\n")

    base_model.trainable = True

    fine_tune_at = len(base_model.layers) - 30

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[checkpoint, early_stop],
    )

    print("\nEvaluating best model...\n")

    model.load_weights(os.path.join(MODEL_DIR, "screen_detector.keras"))

    loss, acc, precision, recall = model.evaluate(val_ds)

    print("\n==============================")
    print(f"Validation Accuracy : {acc:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print("==============================")

    print("\nTraining Complete.")
    print(f"Best model saved to: {MODEL_DIR}/screen_detector.keras")


if __name__ == "__main__":
    main()