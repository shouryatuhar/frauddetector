"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

"""
predict.py

Usage:
    python predict.py some_image.jpg

Prints one number:
0.0 = real photo
1.0 = screen recapture
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from moire_feature import moire_score

IMG_SIZE = (224, 224)

MODEL = tf.keras.models.load_model("model/screen_detector_combined.keras")

MN, MX = np.load("model/moire_norm.npy")

try:
    THRESHOLD = float(np.load("model/best_threshold.npy")[0])
except Exception:
    THRESHOLD = 0.5


def preprocess_pil(img):
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return arr


def center_crop(img):
    w, h = img.size
    s = int(min(w, h) * 0.9)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize(IMG_SIZE)


def predict(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")

    raw = moire_score(image_path)
    moire = (raw - MN) / (MX - MN + 1e-8)
    moire = np.clip(moire, 0.0, 1.0).astype(np.float32)
    feat = np.array([[moire]], dtype=np.float32)

    variants = [
        preprocess_pil(img),
        preprocess_pil(img.transpose(Image.FLIP_LEFT_RIGHT)),
        preprocess_pil(center_crop(img))
    ]

    probs = []

    for v in variants:
        x = np.expand_dims(v, 0)
        p = MODEL.predict([x, feat], verbose=0)[0][0]
        probs.append(float(p))

    return float(np.mean(probs))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py image.jpg")
        sys.exit(1)

    prob = predict(sys.argv[1])
    print(f"{prob:.6f}")
