# 🧠 Migraine Risk Detection using Machine Learning

A final-year B.Tech (CSE) project that predicts a person's **migraine risk level
(Low / Medium / High)** from everyday lifestyle and health factors — sleep,
stress, screen time, hydration, caffeine, family history, hormonal changes,
and more — using a trained ML classifier, wrapped in a Flask web application.

---

## 1. Project Overview

Migraines are influenced by a well-documented set of triggers: irregular
sleep, high stress, dehydration, skipped meals, excessive screen time,
weather sensitivity, hormonal changes, and genetic/family history. This
project:

1. Generates a clinically-inspired **synthetic dataset** of 6,000 patient
   profiles (see `data/generate_dataset.py`) — each record's risk label is
   computed from a weighted combination of known migraine triggers plus
   random noise, so the data is realistic rather than perfectly separable.
2. Trains and compares **three ML models** — Logistic Regression, SVM (RBF
   kernel), and Random Forest — and automatically selects the best performer
   by weighted F1-score (`model/train_model.py`).
3. Serves an interactive **Flask web app** where a user enters their own
   details via sliders/toggles and instantly gets a predicted risk level,
   class probabilities, and lifestyle tips (`app.py` + `templates/`,
   `static/`).

> ⚠️ **Disclaimer:** This is an educational project using synthetic data. It
> is *not* a medical diagnostic tool and must not be used as a substitute for
> professional medical advice.

---

## 2. Project Structure

```
migraine_project/
├── app.py                     # Flask application (routes + prediction API)
├── requirements.txt
├── data/
│   ├── generate_dataset.py    # Synthetic dataset generator
│   └── migraine_dataset.csv   # Generated dataset (6000 rows)
├── model/
│   ├── train_model.py         # Trains, evaluates & saves the ML model
│   ├── migraine_model.pkl     # Trained model (best of 3)
│   ├── scaler.pkl             # StandardScaler used before prediction
│   ├── gender_encoder.pkl     # LabelEncoder for Gender
│   ├── target_encoder.pkl     # LabelEncoder for Risk labels
│   ├── results.json           # Metrics for all trained models
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── model_comparison.png
├── templates/
│   └── index.html             # Web UI
└── static/
    ├── style.css
    ├── script.js
    └── *.png                  # Charts shown on the web page
```

---

## 3. How to Run

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate dataset & retrain model
python data/generate_dataset.py
python model/train_model.py

# 4. Run the web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 4. Features Used by the Model

| Feature | Description |
|---|---|
| Age, Gender | Basic demographics |
| Sleep_Hours | Hours slept the previous night |
| Stress_Level | Self-reported stress (1–10) |
| Screen_Time_Hours | Daily screen exposure |
| Water_Intake_Liters | Daily hydration |
| Skipped_Meals | Meals skipped that day |
| Caffeine_Cups | Cups of caffeinated drinks/day |
| Physical_Activity_Hours | Daily exercise |
| Weather_Sensitive | Headaches triggered by weather change |
| Family_History | Family history of migraine |
| Hormonal_Changes | Hormonal fluctuation (applicable mainly for women) |
| Alcohol_Consumption | Alcohol in the last 24 hrs |
| Noise_Sensitivity, Light_Sensitivity | Sensory sensitivity (1–10) |

---

## 5. Model Performance (on held-out test set)

Run `model/train_model.py` to regenerate these numbers. Typical results:

| Model | Accuracy | Weighted F1 |
|---|---|---|
| Logistic Regression | ~66% | ~66% |
| SVM (RBF Kernel) | ~65% | ~66% |
| Random Forest | ~61% | ~61% |

The 3-class problem (Low / Medium / High) is intentionally noisy — like real
health data — so "Medium" (the boundary class) is the hardest to classify,
which is reflected in the confusion matrix (`model/confusion_matrix.png`).

---

## 6. Possible Extensions (good talking points for viva)

- Replace the synthetic dataset with a real, IRB-approved migraine dataset
  (e.g. from Kaggle/UCI) if available.
- Add a time-series component: predict risk trend over a week using a
  user's daily log (would suit an LSTM/GRU).
- Add user accounts + a migraine diary that feeds back into retraining.
- Deploy on Render/Railway/PythonAnywhere with a Postgres backend instead of
  file-based `.pkl` models.
- Add SHAP-based explainability per-prediction instead of only global
  feature importance.
- Hyperparameter tuning via `GridSearchCV` / `Optuna` for a slightly higher
  ceiling on accuracy.

---

## 7. Tech Stack

- **Backend / ML:** Python, scikit-learn, pandas, NumPy, Flask
- **Frontend:** HTML5, CSS3, vanilla JavaScript (fetch API, no framework)
- **Visualization:** Matplotlib (confusion matrix, feature importance,
  model comparison charts)
