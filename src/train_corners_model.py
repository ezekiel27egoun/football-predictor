"""
Entraîne les modèles de corners attendus (regression) : un pour les corners
domicile, un pour les corners extérieur. Même principe que
train_goals_model.py -- seule la cible change.

Limite connue : aucune donnée de corners pour la Champions League (source
fbref, buts uniquement) -> ces lignes sont naturellement absentes de
l'entraînement (dropna sur la cible), et ne seront pas prédites côté
inférence (cf. predict_matches.py).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

df_features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "matches_features.csv", parse_dates=["date"])

diff_base_cols = [
    c[len("home_"):] for c in df_features.columns
    if c.startswith("home_") and ("_avg_last" in c or c.endswith("matches_played_before"))
]
for col in diff_base_cols:
    df_features[f"diff_{col}"] = df_features[f"home_{col}"] - df_features[f"away_{col}"]

df_model = df_features.sort_values("date").reset_index(drop=True)
# Absent pour la Champions League -> exclue naturellement de l'entraînement
df_model = df_model.dropna(subset=["home_corners", "away_corners"])

df_train, df_temp = train_test_split(df_model, test_size=0.30, shuffle=False)
df_val, df_test = train_test_split(df_temp, test_size=0.50, shuffle=False)

feature_cols = [
    c for c in df_model.columns
    if "_avg_last" in c
    or c.endswith("matches_played_before")
    or c.endswith("days_since_last_match")
    or "h2h" in c
    or c.endswith("elo_before")
    or c.endswith("league_position")
    or c.endswith("season_points_before")
    or c.endswith("domestic_position")
    or c.endswith("prev_seasons_avg_position")
    or c.endswith("reigning_champion")
    or c.endswith("team_tracked")
]

h2h_avg_cols = [c for c in feature_cols if "h2h_points_avg" in c or "h2h_goal_diff_avg" in c]
position_cols = [
    c for c in feature_cols
    if c.endswith("league_position") or c.endswith("season_points_before")
    or c.endswith("domestic_position") or c.endswith("prev_seasons_avg_position")
]
position_col_means = df_train[position_cols].mean()


def make_xy(df, target_col):
    df_clean = df.copy()
    df_clean[h2h_avg_cols] = df_clean[h2h_avg_cols].fillna(0)
    df_clean[position_cols] = df_clean[position_cols].fillna(position_col_means)
    df_clean = df_clean.dropna(subset=feature_cols)
    return df_clean[feature_cols], df_clean[target_col]


def train_one(target_col, label):
    X_train, y_train = make_xy(df_train, target_col)
    X_val, y_val = make_xy(df_val, target_col)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_val, model.predict(X_val))
    print(f"{label} : MAE validation = {mae:.3f} corner(s) "
          f"(moyenne réelle de la cible sur validation : {y_val.mean():.2f})")
    return model


if __name__ == "__main__":
    model_home = train_one("home_corners", "Corners domicile")
    model_away = train_one("away_corners", "Corners extérieur")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model_home, models_dir / "corners_home.joblib")
    joblib.dump(model_away, models_dir / "corners_away.joblib")
    joblib.dump(feature_cols, models_dir / "corners_feature_cols.joblib")
    print(f"\nSauvegardé -> {models_dir / 'corners_home.joblib'}")
    print(f"Sauvegardé -> {models_dir / 'corners_away.joblib'}")
    print(f"Sauvegardé -> {models_dir / 'corners_feature_cols.joblib'}")
