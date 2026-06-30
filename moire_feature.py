"""
Frequency-domain (moire) screen-recapture detector.

Idea: screens have a fixed pixel grid. When re-photographed, the camera's
sensor grid beats against the screen's pixel grid, producing periodic
high-frequency energy (moire) that isn't present in photos of real objects.
We measure this by looking at the FFT magnitude spectrum and scoring how
much energy sits in a "mid-to-high frequency ring" relative to total energy.

This is meant to be tested standalone first (no training required) to see
how separable it makes the two classes, before deciding whether to combine
it with the CNN.

Usage:
    python moire_feature.py        # runs on dataset/real and dataset/screen,
                                    # prints per-class score stats + a simple
                                    # threshold-based accuracy estimate
"""

import os
import numpy as np
from PIL import Image

DATA_DIR = "dataset"
IMG_SIZE = 512  # resize for consistent FFT resolution


def moire_score(image_path: str) -> float:
    """Return a 0-1ish score: higher = more likely a screen recapture."""
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)

    # 2D FFT, shift zero-freq to center
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # build a radial distance map
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cy, cx)

    # ignore very low frequencies (overall brightness/contrast, DC component)
    # and very high frequencies (sensor noise); focus on the "ring" where
    # moire typically shows up
    low_cut = max_r * 0.08
    high_cut = max_r * 0.45

    ring_mask = (r >= low_cut) & (r <= high_cut)
    total_mask = r <= max_r

    ring_energy = magnitude[ring_mask].sum()
    total_energy = magnitude[total_mask].sum() + 1e-8

    score = ring_energy / total_energy
    return float(score)


def evaluate_on_dataset():
    real_dir = os.path.join(DATA_DIR, "real")
    screen_dir = os.path.join(DATA_DIR, "screen")

    real_scores = []
    for f in os.listdir(real_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            real_scores.append(moire_score(os.path.join(real_dir, f)))

    screen_scores = []
    for f in os.listdir(screen_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            screen_scores.append(moire_score(os.path.join(screen_dir, f)))

    real_scores = np.array(real_scores)
    screen_scores = np.array(screen_scores)

    print(f"REAL   -> mean: {real_scores.mean():.4f}  std: {real_scores.std():.4f}  "
          f"min: {real_scores.min():.4f}  max: {real_scores.max():.4f}")
    print(f"SCREEN -> mean: {screen_scores.mean():.4f}  std: {screen_scores.std():.4f}  "
          f"min: {screen_scores.min():.4f}  max: {screen_scores.max():.4f}")

    # sweep thresholds to find the best simple cutoff
    all_scores = np.concatenate([real_scores, screen_scores])
    all_labels = np.concatenate([
        np.zeros(len(real_scores)),
        np.ones(len(screen_scores)),
    ])

    thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)
    best_acc = 0
    best_thresh = None

    for t in thresholds:
        preds = (all_scores >= t).astype(int)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"\nBest single threshold: {best_thresh:.4f}")
    print(f"Accuracy at best threshold (on full set, not held-out): {best_acc:.4f}")
    print("\n(Note: this accuracy is optimistic since the threshold was chosen "
          "on this same data. Re-check on a held-out split before trusting it.)")


if __name__ == "__main__":
    evaluate_on_dataset()