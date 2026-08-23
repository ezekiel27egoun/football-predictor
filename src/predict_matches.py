"""
Pipeline de prédiction : matchs à venir -> probabilités H/D/A.

Principe : on ajoute les matchs à venir comme des lignes "fantômes" (sans
résultat) à la suite de l'historique, on relance exactement le même calcul
de features que pour l'entraînement (shift(1) partout -> aucune fuite de
données), puis on lit les features calculées pour ces lignes fantômes.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cards_markets import compute_cards_markets
from corners_markets import compute_corners_markets
from feature_engineering import build_features
from fetch_fixtures import get_upcoming_fixtures
from goals_markets import compute_goals_markets

# Chemins construits depuis l'emplacement du fichier -> fonctionne peu
# importe le répertoire courant depuis lequel le script/l'app est lancé.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "rf_v8.joblib"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_cols.joblib"
GOALS_HOME_MODEL_PATH = PROJECT_ROOT / "models" / "goals_home.joblib"
GOALS_AWAY_MODEL_PATH = PROJECT_ROOT / "models" / "goals_away.joblib"
CORNERS_HOME_MODEL_PATH = PROJECT_ROOT / "models" / "corners_home.joblib"
CORNERS_AWAY_MODEL_PATH = PROJECT_ROOT / "models" / "corners_away.joblib"
CARDS_YELLOW_HOME_MODEL_PATH = PROJECT_ROOT / "models" / "cards_yellow_home.joblib"
CARDS_YELLOW_AWAY_MODEL_PATH = PROJECT_ROOT / "models" / "cards_yellow_away.joblib"
CARDS_RED_TOTAL_MODEL_PATH = PROJECT_ROOT / "models" / "cards_red_total.joblib"
HISTORICAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_all_raw.csv"


def compute_league_averages(df_features_all, feature_cols):
    """
    Moyenne par ligue de chaque feature, calculée sur les matchs déjà joués
    de df_features_all (celles-ci incluent les colonnes diff_*, absentes du
    CSV matches_features.csv sauvegardé) -> valeur de repli pour une équipe
    sans aucun historique (ex: équipe tout juste promue), plutôt qu'un NaN
    qui ferait planter le modèle.
    """
    df_played = df_features_all[df_features_all["full_time_home_goals"].notna()]
    return df_played.groupby("league")[feature_cols].mean()


def season_label_from_date(dates):
    """
    "season" côté API football-data.org est un identifiant numérique
    (m["season"]["id"]), pas un label "YYYY-YYYY" comme dans l'historique
    -> mélanger les deux dans une même colonne cassait tout calcul qui trie/
    compare les saisons (ex: add_multi_season_strength). Saison européenne =
    juillet année N -> juin année N+1, dérivée directement de la date du match.
    """
    years = dates.dt.year
    start_years = years.where(dates.dt.month >= 7, years - 1)
    return start_years.astype(str) + "-" + (start_years + 1).astype(str)


def build_phantom_rows(fixtures_df):
    """Convertit les fixtures à venir au format attendu par build_features (schéma de df_all)."""
    phantom = pd.DataFrame({
        "date": fixtures_df["date"],
        "league": fixtures_df["league"],
        "season": season_label_from_date(fixtures_df["date"]),
        "home_team": fixtures_df["home_team"],
        "away_team": fixtures_df["away_team"],
        # np.nan (pas pd.NA) : pd.NA force les colonnes en dtype "object" une
        # fois concaténées à l'historique (float64), ce qui casse les .mean()
        # /.rolling() en aval ("No numeric types to aggregate").
        "full_time_home_goals": np.nan,
        "full_time_away_goals": np.nan,
    })
    # Les autres colonnes de stats brutes (tirs, corners...) ne sont pas
    # nécessaires pour les lignes fantômes : elles ne servent qu'à calculer
    # les features des matchs FUTURS, jamais consommées pour elles-mêmes
    # côté fantôme (shift(1) les ignore toujours).
    return phantom


OUTPUT_COLS = [
    "date", "kickoff_utc", "league", "matchday", "home_team_api", "away_team_api",
    "home_crest", "away_crest", "home_team_known", "away_team_known",
    "status", "home_score", "away_score",
    "proba_H", "proba_D", "proba_A",
    "expected_home_goals", "expected_away_goals",
    "proba_home_over_0_5", "proba_home_over_1_5", "proba_home_over_2_5",
    "proba_away_over_0_5", "proba_away_over_1_5", "proba_away_over_2_5",
    "proba_over_1_5", "proba_over_2_5", "proba_over_3_5", "proba_btts_yes",
    "expected_home_corners", "expected_away_corners",
    "proba_home_corners_over_3_5", "proba_home_corners_over_4_5",
    "proba_away_corners_over_3_5", "proba_away_corners_over_4_5",
    "proba_corners_over_8_5", "proba_corners_over_9_5", "proba_corners_over_10_5",
    "expected_home_yellow_cards", "expected_away_yellow_cards",
    "proba_home_yellow_over_1_5", "proba_home_yellow_over_2_5",
    "proba_away_yellow_over_1_5", "proba_away_yellow_over_2_5",
    "proba_yellow_over_3_5", "proba_yellow_over_4_5",
    "expected_red_cards", "proba_red_card_in_match",
]


def predict_upcoming_matches(date_from, date_to):
    """
    Retourne un DataFrame pour tous les matchs (passés ET à venir) de la
    plage demandée :
    - matchs déjà joués (status == "FINISHED") -> score réel renseigné,
      proba_* à NaN (inutile de prédire un résultat déjà connu)
    - matchs pas encore joués -> proba_* calculées par le modèle, score
      à NaN
    """
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    goals_model_home = joblib.load(GOALS_HOME_MODEL_PATH)
    goals_model_away = joblib.load(GOALS_AWAY_MODEL_PATH)
    corners_model_home = joblib.load(CORNERS_HOME_MODEL_PATH)
    corners_model_away = joblib.load(CORNERS_AWAY_MODEL_PATH)
    cards_yellow_model_home = joblib.load(CARDS_YELLOW_HOME_MODEL_PATH)
    cards_yellow_model_away = joblib.load(CARDS_YELLOW_AWAY_MODEL_PATH)
    cards_red_model_total = joblib.load(CARDS_RED_TOTAL_MODEL_PATH)

    fixtures_df = get_upcoming_fixtures(date_from, date_to, status=None)
    if fixtures_df.empty:
        return fixtures_df

    played_mask = fixtures_df["status"] == "FINISHED"
    df_played = fixtures_df[played_mask].copy()
    df_to_predict = fixtures_df[~played_mask].copy()

    derived_out_cols = [
        "expected_home_goals", "expected_away_goals",
        "proba_home_over_0_5", "proba_home_over_1_5", "proba_home_over_2_5",
        "proba_away_over_0_5", "proba_away_over_1_5", "proba_away_over_2_5",
        "proba_over_1_5", "proba_over_2_5", "proba_over_3_5", "proba_btts_yes",
        "expected_home_corners", "expected_away_corners",
        "proba_home_corners_over_3_5", "proba_home_corners_over_4_5",
        "proba_away_corners_over_3_5", "proba_away_corners_over_4_5",
        "proba_corners_over_8_5", "proba_corners_over_9_5", "proba_corners_over_10_5",
        "expected_home_yellow_cards", "expected_away_yellow_cards",
        "proba_home_yellow_over_1_5", "proba_home_yellow_over_2_5",
        "proba_away_yellow_over_1_5", "proba_away_yellow_over_2_5",
        "proba_yellow_over_3_5", "proba_yellow_over_4_5",
        "expected_red_cards", "proba_red_card_in_match",
    ]
    for c in ("proba_H", "proba_D", "proba_A", *derived_out_cols):
        df_played[c] = np.nan

    if df_to_predict.empty:
        return df_played[OUTPUT_COLS].sort_values(["kickoff_utc", "league"])

    df_hist = pd.read_csv(HISTORICAL_DATA_PATH, parse_dates=["date"])
    phantom_rows = build_phantom_rows(df_to_predict)

    df_extended = pd.concat([df_hist, phantom_rows], ignore_index=True)
    df_extended["date"] = pd.to_datetime(df_extended["date"])
    df_extended["full_time_home_goals"] = df_extended["full_time_home_goals"].astype(float)
    df_extended["full_time_away_goals"] = df_extended["full_time_away_goals"].astype(float)
    df_features_all = build_features(df_extended)

    # On ne garde que les lignes fantômes qu'on vient d'ajouter (les matchs à venir)
    df_pred = df_features_all.merge(
        df_to_predict[["date", "kickoff_utc", "home_team", "away_team", "home_team_known", "away_team_known",
                        "matchday", "home_team_api", "away_team_api", "home_crest", "away_crest",
                        "status", "home_score", "away_score"]],
        on=["date", "home_team", "away_team"],
        how="inner",
    )

    # Repli moyenne de ligue pour les NaN restants (équipes sans historique)
    league_averages = compute_league_averages(df_features_all, feature_cols)
    for league in df_pred["league"].unique():
        mask = df_pred["league"] == league
        df_pred.loc[mask, feature_cols] = df_pred.loc[mask, feature_cols].fillna(league_averages.loc[league])

    X_pred = df_pred[feature_cols]
    probas = model.predict_proba(X_pred)

    for i, class_label in enumerate(model.classes_):
        df_pred[f"proba_{class_label}"] = probas[:, i]

    # Buts attendus (regression) -> marchés dérivés (over/under, BTTS) par
    # calcul de Poisson (goals_markets.py), même matrice de features X_pred
    # que le modèle H/D/A (mêmes colonnes -> pas de recalcul nécessaire).
    lambda_home = goals_model_home.predict(X_pred)
    lambda_away = goals_model_away.predict(X_pred)
    for key, values in compute_goals_markets(lambda_home, lambda_away).items():
        df_pred[key] = values

    # Corners attendus -> mêmes principes (Poisson). Pas de données brutes
    # de corners pour la Champions League, mais les clubs qui jouent aussi
    # dans un championnat domestique suivi héritent de leur forme corners
    # via le pooling multi-compétitions (cf. add_rolling_features) -> pas
    # besoin d'exclure la C1, le repli moyenne de ligue gère le reste.
    corners_lambda_home = corners_model_home.predict(X_pred)
    corners_lambda_away = corners_model_away.predict(X_pred)
    for key, values in compute_corners_markets(corners_lambda_home, corners_lambda_away).items():
        df_pred[key] = values

    # Cartons -> mêmes principes (Poisson). Jaunes par équipe, rouges en un
    # seul total match (trop rare pour un découpage par équipe fiable, cf.
    # cards_markets.py). Même absence de données Champions League que pour
    # les corners, gérée de la même façon (repli moyenne de ligue).
    cards_lambda_yellow_home = cards_yellow_model_home.predict(X_pred)
    cards_lambda_yellow_away = cards_yellow_model_away.predict(X_pred)
    cards_lambda_red_total = cards_red_model_total.predict(X_pred)
    for key, values in compute_cards_markets(
        cards_lambda_yellow_home, cards_lambda_yellow_away, cards_lambda_red_total
    ).items():
        df_pred[key] = values

    df_result = pd.concat([df_played[OUTPUT_COLS], df_pred[OUTPUT_COLS]], ignore_index=True)
    return df_result.sort_values(["kickoff_utc", "league"])


if __name__ == "__main__":
    result = predict_upcoming_matches("2026-08-21", "2026-08-24")
    pd.set_option("display.width", 150)
    print(result.to_string(index=False))
