import os
import librosa
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def features(path):
    y, sr = librosa.load(path, sr=16000, duration=8)
    if len(y) < 16000:
        return None
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    c = librosa.feature.spectral_centroid(y=y, sr=sr)
    b = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    z = librosa.feature.zero_crossing_rate(y)
    return np.concatenate([
        np.mean(m, axis=1), np.std(m, axis=1),
        [np.mean(c), np.std(c)],
        [np.mean(b), np.std(b)],
        [np.mean(z), np.std(z)]
    ])


X, y, groups = [], [], []

for label, folder in enumerate(["human", "ai"]):
    directory = os.path.join("dataset", folder)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Папка не найдена: {directory}")

    for filename in os.listdir(directory):
        if filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            path = os.path.join(directory, filename)
            try:
                f = features(path)
                if f is not None:
                    X.append(f)
                    y.append(label)
                    groups.append(filename.split("_")[0])
                    print("OK  ", folder, filename)
                else:
                    print("SKIP", folder, filename, "— короче 1 сек")
            except Exception as e:
                print("SKIP", folder, filename, "—", e)

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print(f"\nВсего: {len(y)} | Human: {int((y==0).sum())} | AI: {int((y==1).sum())}")
print(f"Уникальных групп: {len(set(groups))}")

if len(set(y)) != 2:
    raise ValueError("Нужны и human, и ai.")

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
)

if len(y) >= 10 and len(set(groups)) >= 4:
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(split.split(X, y, groups))
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    print("\n--- Тест: GroupShuffleSplit 20% ---")
    print(classification_report(y[te], pred, labels=[0, 1],
                                target_names=["Human", "AI"], zero_division=0))
elif len(y) >= 10:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("\n--- Тест: train_test_split 20% ---")
    print(classification_report(yte, pred, labels=[0, 1],
                                target_names=["Human", "AI"], zero_division=0))
else:
    print("\nМало данных — обучение на всём датасете без теста (демо-режим).")
    model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/voice_detector.pkl")
print("Model saved → models/voice_detector.pkl")