import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from moire_feature import moire_score

IMG_SIZE = (224, 224)
MODEL_PATH = "model/screen_detector_combined.keras"
NORM_PATH = "model/moire_norm.npy"
DATASET = "dataset"
ERROR_DIR = "errors"

model = tf.keras.models.load_model(MODEL_PATH)

moire_min, moire_max = np.load(NORM_PATH)

os.makedirs(ERROR_DIR, exist_ok=True)
os.makedirs(os.path.join(ERROR_DIR, "false_positive"), exist_ok=True)
os.makedirs(os.path.join(ERROR_DIR, "false_negative"), exist_ok=True)

def clear_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p):
            os.remove(p)

clear_folder(os.path.join(ERROR_DIR, "false_positive"))
clear_folder(os.path.join(ERROR_DIR, "false_negative"))

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

rows = []

for class_name, label in [("real",0),("screen",1)]:
    folder = os.path.join(DATASET,class_name)
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith((".jpg",".jpeg",".png")):
            continue

        full_path = os.path.join(folder, filename)

        image = load_image(full_path)

        fft = moire_score(full_path)
        fft = (fft - moire_min) / (moire_max - moire_min + 1e-8)
        fft = np.clip(fft,0.0,1.0).astype(np.float32)
        fft = np.array([[fft]],dtype=np.float32)

        prob = float(model.predict([image, fft], verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0

        rows.append({
            "filename": filename,
            "path": full_path,
            "true": label,
            "predicted": pred,
            "probability": prob
        })

        if pred != label:
            target = "false_positive" if pred == 1 else "false_negative"
            shutil.copy2(full_path, os.path.join(ERROR_DIR,target,filename))

df = pd.DataFrame(rows)
df.to_csv("predictions.csv", index=False)

y_true = df["true"]
y_pred = df["predicted"]

acc = accuracy_score(y_true,y_pred)
prec = precision_score(y_true,y_pred)
rec = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)
cm = confusion_matrix(y_true,y_pred)

print("\n==============================")
print("Evaluation Results")
print("==============================")
print(f"Total Images : {len(df)}")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"F1 Score     : {f1:.4f}")
print("\nConfusion Matrix")
print(cm)

real = df[df.true==0]
screen = df[df.true==1]

print(f"\nREAL   : {(real.true==real.predicted).sum()}/{len(real)} correct")
print(f"SCREEN : {(screen.true==screen.predicted).sum()}/{len(screen)} correct")

fp = df[(df.true==0)&(df.predicted==1)]
fn = df[(df.true==1)&(df.predicted==0)]

print(f"\nFalse Positives : {len(fp)}")
print(f"False Negatives : {len(fn)}")

mistakes = df[df.true!=df.predicted].copy()
mistakes["confidence"] = mistakes.apply(
    lambda r: r["probability"] if r["predicted"]==1 else 1-r["probability"], axis=1
)
mistakes = mistakes.sort_values("confidence", ascending=False)

print("\nTop 10 Most Confident Mistakes")
for _,r in mistakes.head(10).iterrows():
    print("-"*40)
    print("File       :", r["filename"])
    print("True       :", "SCREEN" if r["true"] else "REAL")
    print("Predicted  :", "SCREEN" if r["predicted"] else "REAL")
    print("Probability:", round(r["probability"],4))
    print("Confidence :", round(r["confidence"],4))

print("\nSaved predictions.csv")
print("Saved errors/false_positive/")
print("Saved errors/false_negative/")
