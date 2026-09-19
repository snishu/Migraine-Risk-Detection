from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ---- Load trained artifacts once at startup ----
model = joblib.load(os.path.join(MODEL_DIR, "migraine_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
gender_encoder = joblib.load(os.path.join(MODEL_DIR, "gender_encoder.pkl"))
target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))

with open(os.path.join(MODEL_DIR, "results.json")) as f:
    MODEL_INFO = json.load(f)

FEATURES = MODEL_INFO["features"]

TIPS = {
    "Low": [
        "Great job! Keep maintaining your current sleep & hydration routine.",
        "Continue regular physical activity to keep stress levels in check.",
        "Stay mindful of screen time during long work sessions."
    ],
    "Medium": [
        "Try to fix a consistent sleep schedule of 7-8 hours.",
        "Reduce caffeine intake, especially after afternoon.",
        "Take short breaks every 45-60 minutes if screen time is high.",
        "Stay hydrated - aim for at least 2.5L of water a day."
    ],
    "High": [
        "Please consult a healthcare professional / neurologist for evaluation.",
        "Prioritize consistent sleep - irregular sleep is a major trigger.",
        "Track your triggers (food, stress, weather) using a migraine diary.",
        "Avoid skipping meals and manage stress with relaxation techniques."
    ]
}


@app.route("/")
def home():
    return render_template("index.html", model_info=MODEL_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        gender_encoded = gender_encoder.transform([data["gender"]])[0]

        row = [
            float(data["age"]),
            gender_encoded,
            float(data["sleep_hours"]),
            int(data["stress_level"]),
            float(data["screen_time"]),
            float(data["water_intake"]),
            int(data["skipped_meals"]),
            int(data["caffeine_cups"]),
            float(data["physical_activity"]),
            int(data["weather_sensitive"]),
            int(data["family_history"]),
            int(data["hormonal_changes"]),
            int(data["alcohol"]),
            int(data["noise_sensitivity"]),
            int(data["light_sensitivity"]),
        ]

        X = np.array(row).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred_encoded = model.predict(X_scaled)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(X_scaled)[0]
            proba = {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(target_encoder.classes_, proba_arr)
            }

        return jsonify({
            "success": True,
            "prediction": pred_label,
            "probabilities": proba,
            "tips": TIPS.get(pred_label, [])
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/model-info")
def model_info():
    return jsonify(MODEL_INFO)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
