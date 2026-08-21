"""
Reconstruit data/processed/matches_features.csv à partir de
data/processed/matches_all_raw.csv, en appliquant build_features()
(feature_engineering.py). À relancer après toute modification du feature
engineering, ou après un rafraîchissement des données historiques, avant de
ré-entraîner le modèle (train_final_model.py).
"""
from pathlib import Path

import pandas as pd

from feature_engineering import build_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_all_raw.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_features.csv"


def main():
    df_all = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df_features = build_features(df_all)
    df_features.to_csv(OUTPUT_PATH, index=False)
    print(f"Sauvegardé : {df_features.shape[0]} lignes, {df_features.shape[1]} colonnes -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
