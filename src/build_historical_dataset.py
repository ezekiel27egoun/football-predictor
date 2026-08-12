"""
Reconstruit data/processed/matches_all_raw.csv à partir de tous les
fichiers bruts présents dans data/raw/ (5 championnats domestiques +
Champions League).

Logique reprise telle quelle de notebooks/01_exploration.ipynb (cellules
0 à 18), extraite ici pour pouvoir être relancée en ligne de commande
(ex: après un rafraîchissement des données via update_historical_data.py)
sans dépendre de l'exécution du notebook.
"""
import glob
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches_all_raw.csv"

DOMESTIC_LEAGUES = {
    "premier_league": RAW_DIR / "premier_league",
    "ligue_1": RAW_DIR / "ligue_1",
    "liga": RAW_DIR / "liga",
    "bundesliga": RAW_DIR / "bundesliga",
    "serie_a": RAW_DIR / "serie_a",
}

UCL_SEASONS = [
    "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2020-2021",
    "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026",
    "2026-2027",
]

RENAME_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "full_time_home_goals",
    "FTAG": "full_time_away_goals",
    "FTR": "full_time_result",
    "HTHG": "half_time_home_goals",
    "HTAG": "half_time_away_goals",
    "HTR": "half_time_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
    "B365H": "odds_home_win",
    "B365D": "odds_draw",
    "B365A": "odds_away_win",
}

STATS_COLS = [
    "half_time_home_goals", "half_time_away_goals", "full_time_result", "half_time_result",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_fouls", "away_fouls", "home_corners", "away_corners",
    "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards",
    "odds_home_win", "odds_draw", "odds_away_win",
]

COMMON_COLS = ["date", "league", "season", "home_team", "away_team",
               "full_time_home_goals", "full_time_away_goals"] + STATS_COLS


def load_league_season(filepath, league_name, season_label):
    """Charge et nettoie un fichier de résultats football-data.co.uk."""
    df = pd.read_csv(filepath, encoding="latin1")

    colonnes_utiles = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "HTHG", "HTAG", "HTR",
        "HS", "AS", "HST", "AST",
        "HF", "AF",
        "HC", "AC",
        "HY", "AY", "HR", "AR",
        "B365H", "B365D", "B365A",
    ]
    colonnes_presentes = [c for c in colonnes_utiles if c in df.columns]
    df = df[colonnes_presentes].copy()

    df["league"] = league_name
    df["season"] = season_label
    return df


def load_ucl_season(filepath, season_label):
    """Charge et nettoie un fichier de résultats Champions League fbref (.xls = HTML)."""
    tables = pd.read_html(filepath, encoding="utf-8")
    df = tables[0]

    df["Home"] = df["Home"].str.replace(r"\s+[a-z]{2,3}$", "", regex=True)
    df["Away"] = df["Away"].str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
    df["season"] = season_label
    return df[["season", "Round", "Date", "Home", "Away", "Score", "Venue", "Referee"]]


def build_domestic_leagues():
    all_leagues = []
    for league_name, folder in DOMESTIC_LEAGUES.items():
        files = glob.glob(f"{folder}/*.csv")
        for filepath in files:
            if os.path.basename(filepath) == "rename_league_files.py":
                continue
            season = os.path.basename(filepath).replace(f"{league_name}_", "").replace(".csv", "")
            try:
                df_season = load_league_season(filepath, league_name, season)
                all_leagues.append(df_season)
            except Exception as e:
                print(f"ERREUR sur {filepath} : {e}")

    df_leagues_all = pd.concat(all_leagues, ignore_index=True)
    df_leagues_all = df_leagues_all.rename(columns=RENAME_MAP)
    return df_leagues_all


def build_champions_league():
    all_ucl = []
    for season in UCL_SEASONS:
        filepath = RAW_DIR / "champions_league" / f"ucl_{season}.xls"
        if not filepath.exists():
            continue
        try:
            df_season = load_ucl_season(str(filepath), season)
            all_ucl.append(df_season)
        except Exception as e:
            print(f"ERREUR sur {filepath} : {e}")

    df_ucl_all = pd.concat(all_ucl, ignore_index=True)

    # On retire les matchs sans score (pas encore joués)
    df_ucl_all = df_ucl_all.dropna(subset=["Score"])
    df_ucl_all = df_ucl_all[df_ucl_all["Score"].str.contains("–", na=False)]

    scores_extracted = df_ucl_all["Score"].str.extract(r"(\d+)\s*(?:\(\d+\))?\s*–\s*(\d+)\s*(?:\(\d+\))?")
    df_ucl_all["ucl_home_goals"] = scores_extracted[0].astype(int)
    df_ucl_all["ucl_away_goals"] = scores_extracted[1].astype(int)

    df_ucl_final = df_ucl_all.rename(columns={
        "Date": "date",
        "Home": "home_team",
        "Away": "away_team",
        "ucl_home_goals": "full_time_home_goals",
        "ucl_away_goals": "full_time_away_goals",
        "Round": "stage",
    })
    df_ucl_final["league"] = "champions_league"

    for col in STATS_COLS:
        df_ucl_final[col] = pd.NA

    return df_ucl_final[COMMON_COLS].copy()


def build_dataset():
    df_leagues_all = build_domestic_leagues()
    df_leagues_final = df_leagues_all[COMMON_COLS].copy()
    df_ucl_final = build_champions_league()

    # Formats de date différents selon la source -> conversion explicite avant fusion.
    df_leagues_final["date"] = pd.to_datetime(df_leagues_final["date"], format="mixed", dayfirst=True)
    df_ucl_final["date"] = pd.to_datetime(df_ucl_final["date"], format="%Y-%m-%d")

    df_all = pd.concat([df_leagues_final, df_ucl_final], ignore_index=True)
    df_all = df_all.sort_values("date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_PATH, index=False)
    print(f"Sauvegardé : {df_all.shape[0]} lignes, {df_all.shape[1]} colonnes -> {OUTPUT_PATH}")
    print(df_all["league"].value_counts())
    return df_all


if __name__ == "__main__":
    build_dataset()
