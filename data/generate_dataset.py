import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000  # number of synthetic patient records

def generate_dataset(n=N):
    age = np.random.randint(15, 60, n)
    gender = np.random.choice(["Female", "Male"], n, p=[0.65, 0.35])  # migraine more common in females

    sleep_hours = np.round(np.random.normal(6.2, 1.4, n), 1)
    sleep_hours = np.clip(sleep_hours, 3, 10)

    stress_level = np.random.randint(1, 11, n)               # 1-10 scale
    screen_time = np.round(np.random.normal(6.5, 2.5, n), 1) # hours/day
    screen_time = np.clip(screen_time, 0.5, 14)

    water_intake = np.round(np.random.normal(2.0, 0.8, n), 1)  # liters/day
    water_intake = np.clip(water_intake, 0.3, 5)

    skipped_meals = np.random.poisson(1.0, n)                  # meals skipped/day
    skipped_meals = np.clip(skipped_meals, 0, 3)

    caffeine_cups = np.random.poisson(1.5, n)
    caffeine_cups = np.clip(caffeine_cups, 0, 6)

    physical_activity = np.round(np.random.exponential(0.8, n), 1)  # hours/day
    physical_activity = np.clip(physical_activity, 0, 3)

    weather_sensitive = np.random.choice([0, 1], n, p=[0.55, 0.45])
    family_history = np.random.choice([0, 1], n, p=[0.6, 0.4])
    hormonal_changes = np.where(
        gender == "Female", np.random.choice([0, 1], n, p=[0.5, 0.5]), 0
    )
    alcohol = np.random.choice([0, 1], n, p=[0.75, 0.25])
    noise_sensitivity = np.random.randint(1, 11, n)
    light_sensitivity = np.random.randint(1, 11, n)

    # ---- Risk score computed from clinically-inspired weighted rules ----
    risk_score = (
        (7 - sleep_hours) * 3.2 +
        stress_level * 2.6 +
        (screen_time > 8).astype(int) * 6 +
        (water_intake < 1.5).astype(int) * 5 +
        skipped_meals * 4 +
        (caffeine_cups > 3).astype(int) * 3 +
        (physical_activity < 0.3).astype(int) * 2 +
        weather_sensitive * 5 +
        family_history * 9 +
        hormonal_changes * 7 +
        alcohol * 3 +
        noise_sensitivity * 1.1 +
        light_sensitivity * 1.1 +
        np.random.normal(0, 8, n)   # noise for realism
    )

    # Normalize to 0-100
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 100

    # Percentile-based buckets -> keeps classes reasonably balanced for training
    low_cut = np.percentile(risk_score, 33)
    high_cut = np.percentile(risk_score, 67)

    def bucket(x):
        if x < low_cut:
            return "Low"
        elif x < high_cut:
            return "Medium"
        else:
            return "High"

    risk_label = np.array([bucket(x) for x in risk_score])

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Sleep_Hours": sleep_hours,
        "Stress_Level": stress_level,
        "Screen_Time_Hours": screen_time,
        "Water_Intake_Liters": water_intake,
        "Skipped_Meals": skipped_meals,
        "Caffeine_Cups": caffeine_cups,
        "Physical_Activity_Hours": physical_activity,
        "Weather_Sensitive": weather_sensitive,
        "Family_History": family_history,
        "Hormonal_Changes": hormonal_changes,
        "Alcohol_Consumption": alcohol,
        "Noise_Sensitivity": noise_sensitivity,
        "Light_Sensitivity": light_sensitivity,
        "Risk_Score": np.round(risk_score, 2),
        "Migraine_Risk": risk_label
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "/home/claude/migraine_project/data/migraine_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset generated: {out_path}")
    print(df["Migraine_Risk"].value_counts())
    print(df.head())
