"""
Train a combined classifier: MobileNetV3Small image features + FFT moire
score, to detect screen recaptures.

Usage:
    python train_combined.py

Dataset structure:
    dataset/
        real/
        screen/

Saves:
    model/screen_detector_combined.keras
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sklearn.model_selection import train_test_split
from PIL import Image

from moire_feature import moire_score  # reuse the function we already tested

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
# Load image paths + precompute moire scores
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


def precompute_moire_scores(paths):
    print("Computing moire scores for all images (one-time cost)...")
    scores = np.array([moire_score(p) for p in paths], dtype=np.float32)
    # normalize to ~0-1 range using min/max from this dataset
    mn, mx = scores.min(), scores.max()
    scores_norm = (scores - mn) / (mx - mn + 1e-8)
    print(f"Moire score range: [{mn:.4f}, {mx:.4f}]")
    return scores_norm, mn, mx


# ----------------------------
# tf.data pipeline with two inputs
# ----------------------------

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)
    return image


def make_dataset(paths, moire_scores, labels, training=False):
    def gen():
        for p, m, l in zip(paths, moire_scores, labels):
            yield p, m, l

    def map_fn(path, moire, label):
        image = load_image(path)
        return (image, moire), label

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, moire_scores.astype(np.float32), np.array(labels, dtype=np.float32))
    )
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(len(paths), seed=SEED)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ----------------------------
# Build combined model
# ----------------------------

def build_model():
    augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomBrightness(0.20),
    layers.RandomContrast(0.20),
    layers.GaussianNoise(0.02),
])

    base_model = MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    image_input = tf.keras.Input(shape=(224, 224, 3), name="image")
    moire_input = tf.keras.Input(shape=(1,), name="moire_score")

    x = augmentation(image_input)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)

    # concatenate the CNN's pooled features with the moire score
    combined = layers.Concatenate()([x, moire_input])

    y = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(combined)

    y = layers.Dropout(0.35)(y)

    y = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(y)

    y = layers.Dropout(0.25)(y)

    outputs = layers.Dense(1, activation="sigmoid")(y)

    model = tf.keras.Model(inputs=[image_input, moire_input], outputs=outputs)
    return model, base_model


# ----------------------------
# Main
# ----------------------------

def main():
    paths, labels = load_image_paths_and_labels()
    moire_scores, mn, mx = precompute_moire_scores(paths)

    # save normalization constants so predict.py can use the same scaling
    np.save(os.path.join(MODEL_DIR, "moire_norm.npy"), np.array([mn, mx]))

    train_paths, val_paths, train_moire, val_moire, train_labels, val_labels = train_test_split(
        paths, moire_scores, labels,
        test_size=0.2, random_state=SEED, stratify=labels,
    )

    print(f"Training images : {len(train_paths)}")
    print(f"Validation images : {len(val_paths)}")

    train_ds = make_dataset(train_paths, train_moire, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_moire, val_labels)

    model, base_model = build_model()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "screen_detector_combined.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )

    print("\n========== Stage 1 ==========")
    print("Training classifier head (frozen backbone)...\n")

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4
),
loss=tf.keras.losses.BinaryCrossentropy(
    label_smoothing=0.05
),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=20,
              callbacks=[checkpoint, early_stop])

    print("\n========== Stage 2 ==========")
    print("Fine-tuning backbone...\n")

    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 45
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
    learning_rate=5e-6,
    weight_decay=1e-4
),
loss=tf.keras.losses.BinaryCrossentropy(
    label_smoothing=0.05
),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              callbacks=[checkpoint, early_stop])

    print("\nEvaluating best model...\n")
    best_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "screen_detector_combined.keras")
    )

    loss, acc, precision, recall = best_model.evaluate(val_ds)

    print("==============================")
    print(f"Validation Accuracy : {acc:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print("==============================")

    from sklearn.metrics import accuracy_score

    print("Searching for best threshold...")

    all_probs = []
    all_labels = []

    for (images, moire), labels in val_ds:
        probs = best_model.predict([images, moire], verbose=0).flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    best_threshold = 0.50
    best_accuracy = 0.0

    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (np.array(all_probs) >= threshold).astype(int)
        score = accuracy_score(all_labels, preds)

        if score > best_accuracy:
            best_accuracy = score
            best_threshold = float(threshold)

    np.save(
        os.path.join(MODEL_DIR, "best_threshold.npy"),
        np.array([best_threshold], dtype=np.float32)
    )

    print("==============================")
    print(f"Best Threshold      : {best_threshold:.2f}")
    print(f"Best Accuracy       : {best_accuracy:.4f}")
    print("==============================")

    print("\nTraining Complete.")
    print(f"Best model saved to: {MODEL_DIR}/screen_detector_combined.keras")


if __name__ == "__main__":

    main()