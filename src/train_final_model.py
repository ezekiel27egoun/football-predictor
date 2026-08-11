"""
Réentraîne et sauvegarde le modèle final rf_v8, avec les hyperparamètres
déjà déterminés dans 02_modeling.ipynb (max_depth=6, min_samples_leaf=50,
class_weight="balanced"). Évite de refaire tourner toute la recherche
d'hyperparamètres (9 modèles) juste pour régénérer le fichier sauvegardé.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

df_features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "matches_features.csv", parse_dates=["date"])

# Features d'écart (cf. 02_modeling.ipynb, cellule "diff_base_cols")
diff_base_cols = [
    c[len("home_"):] for c in df_features.columns
    if c.startswith("home_") and ("_avg_last" in c or c.endswith("matches_played_before"))
]
for col in diff_base_cols:
    df_features[f"diff_{col}"] = df_features[f"home_{col}"] - df_features[f"away_{col}"]

df_model = df_features.sort_values("date").reset_index(drop=True)
df_train, df_temp = train_test_split(df_model, test_size=0.30, shuffle=False)
df_val, df_test = train_test_split(df_temp, test_size=0.50, shuffle=False)


def add_target(df):
    df = df.copy()
    conditions = [
        df["full_time_home_goals"] > df["full_time_away_goals"],
        df["full_time_home_goals"] == df["full_time_away_goals"],
    ]
    df["result"] = np.select(conditions, ["H", "D"], default="A")
    return df


df_train = add_target(df_train)
df_val = add_target(df_val)
df_test = add_target(df_test)

feature_cols = [
    c for c in df_model.columns
    if "_avg_last" in c
    or c.endswith("matches_played_before")
    or c.endswith("days_since_last_match")
    or "h2h" in c
    or c.endswith("elo_before")
]

h2h_avg_cols = [c for c in feature_cols if "h2h_points_avg" in c or "h2h_goal_diff_avg" in c]


def make_xy(df):
    df_clean = df.copy()
    df_clean[h2h_avg_cols] = df_clean[h2h_avg_cols].fillna(0)
    df_clean = df_clean.dropna(subset=feature_cols)
    return df_clean[feature_cols], df_clean["result"]


X_train, y_train = make_xy(df_train)
X_val, y_val = make_xy(df_val)

rf_v8 = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
)
rf_v8.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

print(f"Accuracy validation : {accuracy_score(y_val, rf_v8.predict(X_val)):.3f}")
print(f"f1-macro validation : {f1_score(y_val, rf_v8.predict(X_val), average='macro'):.3f}")
print("(référence attendue : accuracy 0.507, f1-macro 0.482)")

models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)
joblib.dump(rf_v8, models_dir / "rf_v8.joblib")
joblib.dump(feature_cols, models_dir / "feature_cols.joblib")
print(f"\nSauvegardé -> {models_dir / 'rf_v8.joblib'}")
print(f"Sauvegardé -> {models_dir / 'feature_cols.joblib'}")
