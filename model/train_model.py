"""
Train ML models to predict Migraine Risk (Low / Medium / High)
from lifestyle & health features, then save the best model + scaler
+ label encoders for use in the Flask web app.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

DATA_PATH = "/home/claude/migraine_project/data/migraine_dataset.csv"
MODEL_DIR = "/home/claude/migraine_project/model"

FEATURES = [
    "Age", "Gender", "Sleep_Hours", "Stress_Level", "Screen_Time_Hours",
    "Water_Intake_Liters", "Skipped_Meals", "Caffeine_Cups",
    "Physical_Activity_Hours", "Weather_Sensitive", "Family_History",
    "Hormonal_Changes", "Alcohol_Consumption", "Noise_Sensitivity",
    "Light_Sensitivity"
]
TARGET = "Migraine_Risk"


def main():
    df = pd.read_csv(DATA_PATH)

    # Encode Gender
    gender_encoder = LabelEncoder()
    df["Gender"] = gender_encoder.fit_transform(df["Gender"])  # Female=0, Male=1

    # Encode target
    target_encoder = LabelEncoder()
    df["Migraine_Risk_Encoded"] = target_encoder.fit_transform(df[TARGET])
    print("Target classes:", list(target_encoder.classes_))

    X = df[FEATURES]
    y = df["Migraine_Risk_Encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    best_model_name = None
    best_f1 = -1
    best_model = None

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "f1_weighted": round(f1 * 100, 2),
            "cv_mean": round(cv_scores.mean() * 100, 2),
            "cv_std": round(cv_scores.std() * 100, 2),
        }

        print(f"\n=== {name} ===")
        print(f"Test Accuracy: {acc*100:.2f}%  |  Weighted F1: {f1*100:.2f}%")
        print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        print(classification_report(y_test, preds, target_names=target_encoder.classes_))

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n>>> Best model: {best_model_name} (weighted F1 = {best_f1*100:.2f}%)")

    # Confusion matrix for best model
    best_preds = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, best_preds)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Purples")
    classes = target_encoder.classes_
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12, fontweight="bold")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=140)
    plt.close()

    # Feature importance: use native importances if available (tree models),
    # otherwise fall back to mean absolute coefficient magnitude (linear/SVM models)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_).mean(axis=0)
    else:
        importances = None

    if importances is not None:
        idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([FEATURES[i] for i in idx][::-1], [importances[i] for i in idx][::-1], color="#7c3aed")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance - {best_model_name}")
        plt.tight_layout()
        plt.savefig(f"{MODEL_DIR}/feature_importance.png", dpi=140)
        plt.close()

    # Save model comparison bar chart
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    ax.bar(names, accs, color=["#a78bfa", "#8b5cf6", "#6d28d9"])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 100)
    for i, v in enumerate(accs):
        ax.text(i, v + 1, f"{v}%", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/model_comparison.png", dpi=140)
    plt.close()

    # Persist everything the web app needs
    joblib.dump(best_model, f"{MODEL_DIR}/migraine_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(gender_encoder, f"{MODEL_DIR}/gender_encoder.pkl")
    joblib.dump(target_encoder, f"{MODEL_DIR}/target_encoder.pkl")

    with open(f"{MODEL_DIR}/results.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "results": results,
            "features": FEATURES
        }, f, indent=2)

    print("\nSaved model, scaler, encoders, and charts to /model directory.")


if __name__ == "__main__":
    main()
