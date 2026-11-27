import os

# --- УКАЗЫВАЕМ ПУТЬ К FFMPEG (ДО ИМПОРТА PYDUB) ---
os.environ["PATH"] += os.pathsep + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools", "ffmpeg"))
# -------------------------------------------------

import uuid
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/voice_detector.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def extract_features(path):
    audio, sr = librosa.load(path, sr=16000, duration=8)
    if len(audio) < 16000:
        raise ValueError("Recording must be at least 1 second.")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    return np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        [np.mean(centroid), np.std(centroid)],
        [np.mean(bandwidth), np.std(bandwidth)],
        [np.mean(zcr), np.std(zcr)]
    ]).reshape(1, -1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if model is None:
        return jsonify({"error": "ML model is not trained. Run train.py."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "Audio file not found"}), 400

    file = request.files["audio"]
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".webm"
    path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + ext)
    wav_path = path + ".wav"

    try:
        file.save(path)
        try:
            audio = AudioSegment.from_file(path)
            audio.export(wav_path, format="wav")
            process_path = wav_path
        except Exception as e:
            process_path = path

        features = extract_features(process_path)
        probabilities = model.predict_proba(features)[0]

        human = float(probabilities[0])
        ai = float(probabilities[1])

        return jsonify({
            "prediction": "AI-generated" if ai > human else "Human",
            "ai": round(ai * 100, 1),
            "human": round(human * 100, 1),
            "warning": "Experimental model. Results are not proof of authenticity."
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 400
    finally:
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    app.run(debug=True)