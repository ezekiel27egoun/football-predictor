"""
Suivi hebdomadaire de la performance du modèle : compare les probabilités de
rf_v8 aux résultats réels et aux cotes de bookmaker (The Odds API), semaine
après semaine.

Outil de MESURE uniquement — ne modifie ni ne ré-entraîne le modèle. Sert à
décider ensuite (ailleurs, à la main) s'il faut ré-entraîner plus souvent,
ajouter des features, calibrer les probabilités, etc.

Depuis cette version, le fichier suit aussi les marchés Buts (over/under 2,5,
BTTS) et Corners (over/under 9,5) — comparés uniquement aux résultats réels
(pas de cotes bookmaker disponibles pour ces marchés dans ce pipeline, voir
plus bas). Les corners réels viennent d'une source différente (historique
football-data.co.uk, pas l'API live football-data.org) -> ils ne sont
disponibles que si data/processed/matches_all_raw.csv a été rafraîchi
récemment (update_historical_data.py) ; sinon la colonne reste vide, ce
qui n'empêche pas le reste du suivi (best-effort).

Usage :
    python weekly_tracking.py --predict            # avant la journée : génère
                                                     # data/tracking/predictions_{date}.csv
    python weekly_tracking.py --refresh-odds        # en milieu de semaine : complète les
                                                     # cotes pas encore postées au moment du --predict
    python weekly_tracking.py --update              # après la journée : complète
                                                     # les résultats réels + métriques
                                                     # de tous les fichiers en attente
    python weekly_tracking.py --daily-report        # affiche, jour par jour, le nombre de
                                                     # pronostics corrects (modèle ET bookmaker
                                                     # pour le 1N2 ; modèle seul pour buts/corners)
"""
import argparse
import os
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import log_loss

from build_team_mapping import best_match
from fetch_fixtures import get_upcoming_fixtures, load_team_mapping
from predict_matches import predict_upcoming_matches

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = PROJECT_ROOT / "data" / "tracking"
WEEKLY_SUMMARY_PATH = TRACKING_DIR / "weekly_summary.csv"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
OUTCOMES = ["H", "D", "A"]
# sklearn.metrics.log_loss exige que les colonnes de proba passées soient
# dans l'ordre alphabétique des labels, quel que soit l'ordre de `labels=`
# -> ordre dédié, distinct de OUTCOMES (H/D/A, plus lisible partout ailleurs).
LOG_LOSS_LABELS = sorted(OUTCOMES)  # ["A", "D", "H"]

# Mots-clés pour retrouver le bon sport_key The Odds API (liste dynamique via
# /v4/sports -> pas de sport_key codé en dur, ils peuvent changer/varier).
LEAGUE_ODDS_KEYWORDS = {
    "premier_league": ["epl"],
    "ligue_1": ["ligue_one", "ligue1", "france"],
    "liga": ["la_liga", "spain"],
    "bundesliga": ["bundesliga"],
    "serie_a": ["serie_a", "italy"],
    "champions_league": ["champs_league", "uefa"],
}

PREDICTIONS_FILENAME_RE = re.compile(r"predictions_(\d{4}-\d{2}-\d{2})\.csv$")


# ------------------------------------------------------------------
# 1. Cotes bookmaker (The Odds API)
# ------------------------------------------------------------------

def get_odds_api_key():
    # .strip() : même précaution que pour FOOTBALL_DATA_API_TOKEN -> un
    # secret copié-collé (GitHub Actions, .env) peut embarquer un retour à
    # la ligne en trop.
    key = os.environ.get("ODDS_API_KEY", "").strip() or None
    if not key:
        print("ATTENTION : ODDS_API_KEY absente du .env -> les colonnes de cotes resteront vides.")
    return key


def fetch_sports(api_key):
    """Liste des sports/compétitions disponibles côté The Odds API (pour retrouver le sport_key)."""
    resp = requests.get(f"{ODDS_API_BASE}/sports", params={"apiKey": api_key}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_sport_key(league_name, sports_list):
    """Retrouve le sport_key The Odds API correspondant à une de nos ligues, par mots-clés."""
    keywords = LEAGUE_ODDS_KEYWORDS.get(league_name, [])
    for sport in sports_list:
        key_lower = sport.get("key", "").lower()
        if sport.get("group") == "Soccer" and any(kw in key_lower for kw in keywords):
            return sport["key"]
    return None


def fetch_odds_for_sport(sport_key, api_key, regions="eu", markets="h2h"):
    """Retourne la liste brute des événements + cotes pour un sport_key donné."""
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{sport_key}/odds",
        params={"apiKey": api_key, "regions": regions, "markets": markets, "oddsFormat": "decimal"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def average_event_odds(event):
    """
    Moyenne les cotes décimales H/D/A entre tous les bookmakers d'un événement.
    Retourne (cote_home, cote_draw, cote_away), NaN si un côté est introuvable.
    """
    home_prices, draw_prices, away_prices = [], [], []
    home_name, away_name = event.get("home_team"), event.get("away_team")

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name, price = outcome.get("name"), outcome.get("price")
                if name == home_name:
                    home_prices.append(price)
                elif name == away_name:
                    away_prices.append(price)
                elif name == "Draw":
                    draw_prices.append(price)

    return (
        float(np.mean(home_prices)) if home_prices else np.nan,
        float(np.mean(draw_prices)) if draw_prices else np.nan,
        float(np.mean(away_prices)) if away_prices else np.nan,
    )


def match_events_to_fixtures(events, fixtures_league_df, score_threshold=0.6):
    """
    Associe chaque événement Odds API (noms d'équipes potentiellement différents)
    à une ligne de fixtures_league_df, par fuzzy-match domicile + extérieur.
    Retourne un dict {index_fixture: (cote_home, cote_draw, cote_away)}.
    """
    result = {}
    candidates_home = fixtures_league_df["home_team_api"].tolist()
    candidates_away = fixtures_league_df["away_team_api"].tolist()

    for event in events:
        home_match, home_score = best_match(event.get("home_team", ""), candidates_home)
        away_match, away_score = best_match(event.get("away_team", ""), candidates_away)
        if home_score < score_threshold or away_score < score_threshold:
            continue

        rows = fixtures_league_df[
            (fixtures_league_df["home_team_api"] == home_match)
            & (fixtures_league_df["away_team_api"] == away_match)
        ]
        if rows.empty:
            continue

        result[rows.index[0]] = average_event_odds(event)

    return result


def attach_odds(df_predictions, api_key):
    """
    Ajoute cote_home/cote_draw/cote_away à df_predictions (colonnes home_team_api/
    away_team_api/league déjà présentes). Best-effort : une ligue dont l'appel
    API échoue reste juste à NaN, ça n'interrompt pas le reste.
    """
    df_predictions["cote_home"] = np.nan
    df_predictions["cote_draw"] = np.nan
    df_predictions["cote_away"] = np.nan

    if not api_key:
        return df_predictions

    try:
        sports_list = fetch_sports(api_key)
    except requests.RequestException as e:
        print(f"ATTENTION : impossible de récupérer la liste des sports Odds API ({e}) -> pas de cotes.")
        return df_predictions

    for league_name in df_predictions["league"].unique():
        sport_key = get_sport_key(league_name, sports_list)
        if sport_key is None:
            print(f"[{league_name}] Aucun sport_key Odds API trouvé -> pas de cotes.")
            continue

        try:
            events = fetch_odds_for_sport(sport_key, api_key)
        except requests.RequestException as e:
            print(f"[{league_name}] Échec de la récupération des cotes ({e}) -> pas de cotes.")
            continue

        fixtures_league_df = df_predictions[df_predictions["league"] == league_name]
        matched = match_events_to_fixtures(events, fixtures_league_df)
        for idx, (cote_h, cote_d, cote_a) in matched.items():
            df_predictions.loc[idx, ["cote_home", "cote_draw", "cote_away"]] = [cote_h, cote_d, cote_a]

        print(f"[{league_name}] Cotes trouvées pour {len(matched)}/{len(fixtures_league_df)} match(s).")

    return df_predictions


def normalize_odds(cote_h, cote_d, cote_a):
    """
    Convertit des cotes décimales en probabilités implicites, marge du
    bookmaker (overround) retirée : proba_normalisee = (1/cote) / somme(1/cotes).
    Retourne (nan, nan, nan) si une cote manque.
    """
    if pd.isna(cote_h) or pd.isna(cote_d) or pd.isna(cote_a):
        return np.nan, np.nan, np.nan
    inv = np.array([1 / cote_h, 1 / cote_d, 1 / cote_a])
    total = inv.sum()
    if total == 0:
        return np.nan, np.nan, np.nan
    p_h, p_d, p_a = inv / total
    return p_h, p_d, p_a


# ------------------------------------------------------------------
# 2. Génération des prédictions (--predict)
# ------------------------------------------------------------------

def run_predict(date_from, date_to):
    print(f"Récupération des matchs à venir + prédictions du modèle ({date_from} -> {date_to})...")
    df = predict_upcoming_matches(str(date_from), str(date_to))
    if df.empty:
        print("Aucun match trouvé sur cette période -> rien à sauvegarder.")
        return None

    # predict_upcoming_matches retourne aussi les matchs déjà joués (proba_* à
    # NaN) -> on ne garde que ceux à venir, qui sont l'objet du tracking.
    df = df[df["proba_H"].notna()].copy()
    if df.empty:
        print("Tous les matchs de la période sont déjà joués -> rien à prédire.")
        return None

    api_key = get_odds_api_key()
    df = attach_odds(df, api_key)

    df["prediction_modele"] = df[["proba_H", "proba_D", "proba_A"]].idxmax(axis=1).str.replace("proba_", "")
    # Over/Under, BTTS -> "prédiction" = le côté que le modèle juge le plus
    # probable (>50%), comme prediction_modele pour le 1N2.
    df["prediction_over_2_5"] = np.where(df["proba_over_2_5"] > 0.5, "Over", "Under")
    df["prediction_btts"] = np.where(df["proba_btts_yes"] > 0.5, "Oui", "Non")
    df["prediction_corners_over_9_5"] = np.where(df["proba_corners_over_9_5"] > 0.5, "Over", "Under")

    df_out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "league": df["league"],
        "home_team": df["home_team_api"],
        "away_team": df["away_team_api"],
        "proba_home": df["proba_H"],
        "proba_draw": df["proba_D"],
        "proba_away": df["proba_A"],
        "prediction_modele": df["prediction_modele"],
        "cote_home": df["cote_home"],
        "cote_draw": df["cote_draw"],
        "cote_away": df["cote_away"],
        "resultat_reel": pd.NA,
        "prediction_correcte": pd.NA,
        "prediction_bookmaker": pd.NA,
        "bookmaker_correcte": pd.NA,
        # Buts (over/under 2,5 + BTTS) -> comparés au résultat réel seulement,
        # pas de cotes bookmaker suivies pour ce marché dans ce pipeline.
        "proba_over_2_5": df["proba_over_2_5"],
        "prediction_over_2_5": df["prediction_over_2_5"],
        "resultat_over_2_5": pd.NA,
        "over_2_5_correcte": pd.NA,
        "proba_btts_yes": df["proba_btts_yes"],
        "prediction_btts": df["prediction_btts"],
        "resultat_btts": pd.NA,
        "btts_correcte": pd.NA,
        # Corners (over/under 9,5) -> réel venant d'une source séparée
        # (historique football-data.co.uk), voir attach_real_corners().
        "proba_corners_over_9_5": df["proba_corners_over_9_5"],
        "prediction_corners_over_9_5": df["prediction_corners_over_9_5"],
        "resultat_corners_over_9_5": pd.NA,
        "corners_over_9_5_correcte": pd.NA,
    })

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRACKING_DIR / f"predictions_{date.today().isoformat()}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Sauvegardé : {len(df_out)} match(s) -> {out_path}")
    return out_path


# ------------------------------------------------------------------
# 2bis. Rafraîchissement des cotes manquantes (--refresh-odds)
# ------------------------------------------------------------------
# Certains bookmakers ne postent leurs cotes que 2-4 jours avant le match ->
# une partie des lignes de predictions_{date}.csv générées le lundi (fenêtre
# de 7 jours) ont encore cote_* = NaN à ce moment. Ce mode retente UNIQUEMENT
# les lignes concernées, sans regénérer les prédictions ni retoucher aux
# lignes déjà résolues (déjà jouées).

def refresh_odds_for_file(path):
    df = pd.read_csv(path, parse_dates=["date"])
    missing = df["cote_home"].isna() & df["resultat_reel"].isna()
    if not missing.any():
        print(f"{path.name} : aucune cote manquante à compléter.")
        return df

    subset = df.loc[missing].copy()
    subset = subset.rename(columns={"home_team": "home_team_api", "away_team": "away_team_api"})
    subset = subset.drop(columns=["cote_home", "cote_draw", "cote_away"])

    api_key = get_odds_api_key()
    subset = attach_odds(subset, api_key)

    n_filled = int(subset[["cote_home", "cote_draw", "cote_away"]].notna().any(axis=1).sum())
    df.loc[missing, ["cote_home", "cote_draw", "cote_away"]] = subset[
        ["cote_home", "cote_draw", "cote_away"]
    ].to_numpy()

    df.to_csv(path, index=False)
    print(f"{path.name} : {n_filled}/{int(missing.sum())} cote(s) manquante(s) complétée(s).")
    return df


def run_refresh_odds_all():
    files = sorted(TRACKING_DIR.glob("predictions_*.csv"))
    if not files:
        print("Aucun fichier data/tracking/predictions_*.csv trouvé.")
        return

    for path in files:
        refresh_odds_for_file(path)


# ------------------------------------------------------------------
# 3. Mise à jour avec les résultats réels (--update)
# ------------------------------------------------------------------

def fetch_real_results(date_from, date_to):
    """Résultats réels (football-data.org, status=FINISHED) sur la plage donnée."""
    df = get_upcoming_fixtures(str(date_from), str(date_to), status="FINISHED")
    return df


def result_letter(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return pd.NA
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def bookmaker_pick(cote_home, cote_draw, cote_away):
    """
    "Pronostic" du bookmaker = l'issue à la cote la plus basse (= probabilité
    implicite la plus haute) -> permet de compter ses bons pronostics au même
    titre que ceux du modèle (prediction_modele), pas juste comparer des
    métriques abstraites (log loss, brier).
    """
    if pd.isna(cote_home) or pd.isna(cote_draw) or pd.isna(cote_away):
        return pd.NA
    return min([("H", cote_home), ("D", cote_draw), ("A", cote_away)], key=lambda x: x[1])[0]


# Colonnes ajoutées après la création des tout premiers fichiers de suivi ->
# absentes des anciens CSV, à réintroduire (vides) avant de les manipuler.
NEW_TRACKING_COLS = [
    "prediction_bookmaker", "bookmaker_correcte",
    "proba_over_2_5", "prediction_over_2_5", "resultat_over_2_5", "over_2_5_correcte",
    "proba_btts_yes", "prediction_btts", "resultat_btts", "btts_correcte",
    "proba_corners_over_9_5", "prediction_corners_over_9_5",
    "resultat_corners_over_9_5", "corners_over_9_5_correcte",
]


def fetch_real_corners(date_from, date_to):
    """
    Vrais corners (home_corners/away_corners) des matchs déjà joués, depuis
    l'historique football-data.co.uk (data/processed/matches_all_raw.csv) ->
    PAS depuis football-data.org (l'API live utilisée pour les scores
    n'expose pas les corners). Best-effort : vide si le fichier n'a pas été
    rafraîchi récemment (update_historical_data.py) pour ces dates -> les
    lignes concernées restent simplement sans corners réels, sans bloquer
    le reste de la mise à jour.
    """
    path = PROJECT_ROOT / "data" / "processed" / "matches_all_raw.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "league", "home_team", "away_team", "home_corners", "away_corners"])
    hist = pd.read_csv(path, parse_dates=["date"])
    mask = (hist["date"] >= pd.Timestamp(date_from)) & (hist["date"] <= pd.Timestamp(date_to))
    hist = hist[mask].dropna(subset=["home_corners", "away_corners"])
    return hist[["date", "league", "home_team", "away_team", "home_corners", "away_corners"]]


def attach_real_corners(df, pending_mask):
    """
    Complète resultat_corners_over_9_5/corners_over_9_5_correcte pour les
    lignes de `df` marquées `pending_mask`, en associant chaque match (noms
    d'équipe API) au nom historique via le mapping déjà utilisé ailleurs dans
    le pipeline (data/team_name_mapping.csv), puis en cherchant ce nom dans
    l'historique football-data.co.uk. Retourne le nombre de lignes complétées.
    """
    to_fill = pending_mask & df["resultat_corners_over_9_5"].isna()
    if not to_fill.any():
        return 0

    date_min, date_max = df.loc[to_fill, "date"].min(), df.loc[to_fill, "date"].max()
    corners_hist = fetch_real_corners(date_min, date_max)
    if corners_hist.empty:
        return 0

    mapping = load_team_mapping()
    n_filled = 0
    for idx in df.index[to_fill]:
        row = df.loc[idx]
        home_hist = mapping.get((row["league"], row["home_team"]), row["home_team"])
        away_hist = mapping.get((row["league"], row["away_team"]), row["away_team"])
        match = corners_hist[
            (corners_hist["date"] == row["date"])
            & (corners_hist["league"] == row["league"])
            & (corners_hist["home_team"] == home_hist)
            & (corners_hist["away_team"] == away_hist)
        ]
        if match.empty:
            continue
        total_corners = match.iloc[0]["home_corners"] + match.iloc[0]["away_corners"]
        real_side = "Over" if total_corners > 9.5 else "Under"
        df.at[idx, "resultat_corners_over_9_5"] = real_side
        if pd.notna(row.get("prediction_corners_over_9_5")):
            df.at[idx, "corners_over_9_5_correcte"] = real_side == row["prediction_corners_over_9_5"]
        n_filled += 1
    return n_filled


def _backfill_bookmaker(df):
    """Calcule prediction_bookmaker/bookmaker_correcte pour les lignes déjà
    résolues (resultat_reel connu) qui ont une cote mais n'ont pas encore ce
    calcul (typiquement : résolues avant l'ajout de cette fonctionnalité).
    Retourne le nombre de lignes complétées."""
    mask = df["resultat_reel"].notna() & df["cote_home"].notna() & df["bookmaker_correcte"].isna()
    if not mask.any():
        return 0
    df.loc[mask, "prediction_bookmaker"] = df.loc[mask].apply(
        lambda r: bookmaker_pick(r["cote_home"], r["cote_draw"], r["cote_away"]), axis=1
    )
    df.loc[mask, "bookmaker_correcte"] = df.loc[mask, "resultat_reel"] == df.loc[mask, "prediction_bookmaker"]
    return int(mask.sum())


def run_update_file(path):
    df = pd.read_csv(path, parse_dates=["date"])
    for col in NEW_TRACKING_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    # Tant qu'aucun résultat n'est connu, ces colonnes sont 100% NaN -> pandas
    # les lit en float64. On les repasse en object avant d'y écrire des
    # lettres ('H'/'D'/'A'), 'Over'/'Under'/'Oui'/'Non' ou des booléens.
    object_cols = ["resultat_reel", "prediction_correcte", "prediction_bookmaker", "bookmaker_correcte"] + [
        c for c in NEW_TRACKING_COLS if c.startswith("resultat_") or c.endswith("_correcte")
    ]
    for col in set(object_cols):
        df[col] = df[col].astype("object")

    # Rétro-complétion : des lignes déjà résolues AVANT l'ajout de ces
    # colonnes (bookmaker, corners) ont resultat_reel connu mais rien
    # calculé pour elles -> à rattraper à chaque appel, même quand il n'y a
    # aucun nouveau résultat à aller chercher (sinon jamais recalculé).
    n_backfilled = _backfill_bookmaker(df)
    n_backfilled += attach_real_corners(df, df["resultat_reel"].notna())

    pending = df["resultat_reel"].isna()
    if not pending.any():
        if n_backfilled:
            df.to_csv(path, index=False)
            print(f"{path.name} : déjà complet, {n_backfilled} pronostic(s) rétro-complété(s) (bookmaker/corners).")
        else:
            print(f"{path.name} : déjà complet, rien à faire.")
        return df

    date_min, date_max = df.loc[pending, "date"].min(), df.loc[pending, "date"].max()
    real_results = fetch_real_results(date_min.date(), date_max.date())

    if real_results.empty:
        if n_backfilled:
            df.to_csv(path, index=False)
            print(f"{path.name} : aucun nouveau résultat, {n_backfilled} pronostic(s) rétro-complété(s) (bookmaker/corners).")
        else:
            print(f"{path.name} : aucun résultat disponible pour l'instant.")
        return df

    # get_upcoming_fixtures retourne À LA FOIS home_team (nom historique mappé)
    # ET home_team_api (nom API) -> sélectionner AVANT de renommer, sinon les
    # deux colonnes finissent avec le même nom "home_team" (dupliqué), ce qui
    # fait planter le merge plus bas ("column label is not unique").
    real_results = real_results[["date", "league", "home_team_api", "away_team_api", "home_score", "away_score"]]
    real_results = real_results.rename(columns={"home_team_api": "home_team", "away_team_api": "away_team"})

    df = df.merge(real_results, on=["date", "league", "home_team", "away_team"], how="left", suffixes=("", "_new"))

    newly_resolved = df["home_score"].notna()

    # --- 1N2 : modèle ---
    df.loc[newly_resolved, "resultat_reel"] = df.loc[newly_resolved].apply(
        lambda r: result_letter(r["home_score"], r["away_score"]), axis=1
    )
    df.loc[newly_resolved, "prediction_correcte"] = (
        df.loc[newly_resolved, "resultat_reel"] == df.loc[newly_resolved, "prediction_modele"]
    )

    # --- 1N2 : bookmaker (même résultat réel, comparé à la cote la plus basse) ---
    has_odds = newly_resolved & df["cote_home"].notna()
    df.loc[has_odds, "prediction_bookmaker"] = df.loc[has_odds].apply(
        lambda r: bookmaker_pick(r["cote_home"], r["cote_draw"], r["cote_away"]), axis=1
    )
    df.loc[has_odds, "bookmaker_correcte"] = (
        df.loc[has_odds, "resultat_reel"] == df.loc[has_odds, "prediction_bookmaker"]
    )

    # --- Buts : over/under 2,5 + BTTS (à partir des mêmes scores réels) ---
    total_goals = df.loc[newly_resolved, "home_score"] + df.loc[newly_resolved, "away_score"]
    df.loc[newly_resolved, "resultat_over_2_5"] = np.where(total_goals > 2.5, "Over", "Under")
    has_goals_pred = newly_resolved & df["prediction_over_2_5"].notna()
    df.loc[has_goals_pred, "over_2_5_correcte"] = (
        df.loc[has_goals_pred, "resultat_over_2_5"] == df.loc[has_goals_pred, "prediction_over_2_5"]
    )

    btts_yes = (df.loc[newly_resolved, "home_score"] > 0) & (df.loc[newly_resolved, "away_score"] > 0)
    df.loc[newly_resolved, "resultat_btts"] = np.where(btts_yes, "Oui", "Non")
    has_btts_pred = newly_resolved & df["prediction_btts"].notna()
    df.loc[has_btts_pred, "btts_correcte"] = (
        df.loc[has_btts_pred, "resultat_btts"] == df.loc[has_btts_pred, "prediction_btts"]
    )

    df = df.drop(columns=["home_score", "away_score"])

    # --- Corners : over/under 9,5 (source séparée, best-effort) ---
    n_corners_filled = attach_real_corners(df, newly_resolved)

    # newly_resolved peut re-toucher des lignes déjà résolues avant cet appel
    # (la fenêtre de dates fetchée n'est pas limitée aux seules lignes pending) ->
    # ne compter comme "nouveau" que les lignes réellement passées de pending à
    # résolu, et calculer le nombre encore en attente sur l'état final, pas par
    # simple soustraction (source du bug : comptage faux si des lignes déjà
    # résolues sont retouchées sans rien changer).
    still_pending = df["resultat_reel"].isna()
    n_new = int((pending & ~still_pending).sum())

    df.to_csv(path, index=False)
    print(
        f"{path.name} : {n_new} résultat(s) 1N2 mis à jour "
        f"({int(still_pending.sum())} encore en attente), {n_corners_filled} corner(s) complété(s)."
    )
    return df


# ------------------------------------------------------------------
# 4. Métriques de comparaison modèle vs bookmaker
# ------------------------------------------------------------------

def brier_score_multiclass(proba_df, actual_series):
    """
    Brier score multi-classe = moyenne, sur les matchs, de la somme des
    (proba_k - reel_k)^2 pour k in {H, D, A} (reel_k = 1 si k est le résultat
    réel, 0 sinon). sklearn ne propose que le cas binaire, implémenté à la main.
    """
    # reindex (pas indexation directe) : si une issue (ex "A") n'apparaît dans
    # aucun match résolu de la semaine, get_dummies ne crée pas sa colonne ->
    # reindex la rajoute à 0 au lieu de lever un KeyError.
    onehot = pd.get_dummies(actual_series).reindex(columns=OUTCOMES, fill_value=0).to_numpy(dtype=float)
    probas = proba_df[OUTCOMES].to_numpy(dtype=float)
    return float(np.mean(np.sum((probas - onehot) ** 2, axis=1)))


def compute_week_metrics(df):
    """
    Calcule les métriques de la semaine à partir des lignes qui ont un
    résultat réel connu. Les métriques bookmaker restent NaN si les cotes
    manquent sur trop de lignes.
    """
    resolved = df[df["resultat_reel"].notna()].copy()
    metrics = {
        "n_matchs_resolus": len(resolved),
        "accuracy_modele": np.nan,
        "log_loss_modele": np.nan,
        "log_loss_bookmaker": np.nan,
        "brier_modele": np.nan,
        "brier_bookmaker": np.nan,
        # Comptages "X bons pronostics sur Y" -> la preuve concrète demandée,
        # en plus des métriques abstraites (log loss/brier) ci-dessus.
        "n_correct_modele": 0,
        "n_avec_cote": 0,
        "n_correct_bookmaker": 0,
        "n_over_2_5": 0,
        "n_correct_over_2_5": 0,
        "n_btts": 0,
        "n_correct_btts": 0,
        "n_corners_9_5": 0,
        "n_correct_corners_9_5": 0,
    }
    if resolved.empty:
        return metrics

    metrics["accuracy_modele"] = float((resolved["prediction_modele"] == resolved["resultat_reel"]).mean())
    metrics["n_correct_modele"] = int((resolved["prediction_modele"] == resolved["resultat_reel"]).sum())

    proba_modele = resolved.rename(columns={"proba_home": "H", "proba_draw": "D", "proba_away": "A"})
    metrics["log_loss_modele"] = log_loss(
        resolved["resultat_reel"], proba_modele[LOG_LOSS_LABELS], labels=LOG_LOSS_LABELS
    )
    metrics["brier_modele"] = brier_score_multiclass(proba_modele, resolved["resultat_reel"])

    normalized = resolved.apply(
        lambda r: normalize_odds(r["cote_home"], r["cote_draw"], r["cote_away"]), axis=1, result_type="expand"
    )
    normalized.columns = ["H", "D", "A"]
    with_odds = normalized.dropna()
    if not with_odds.empty:
        actual_with_odds = resolved.loc[with_odds.index, "resultat_reel"]
        metrics["log_loss_bookmaker"] = log_loss(
            actual_with_odds, with_odds[LOG_LOSS_LABELS], labels=LOG_LOSS_LABELS
        )
        metrics["brier_bookmaker"] = brier_score_multiclass(with_odds, actual_with_odds)

    if "bookmaker_correcte" in resolved.columns:
        with_bookmaker_pick = resolved["bookmaker_correcte"].notna()
        metrics["n_avec_cote"] = int(with_bookmaker_pick.sum())
        metrics["n_correct_bookmaker"] = int(resolved.loc[with_bookmaker_pick, "bookmaker_correcte"].sum())

    if "over_2_5_correcte" in resolved.columns:
        m = resolved["over_2_5_correcte"].notna()
        metrics["n_over_2_5"] = int(m.sum())
        metrics["n_correct_over_2_5"] = int(resolved.loc[m, "over_2_5_correcte"].sum())

    if "btts_correcte" in resolved.columns:
        m = resolved["btts_correcte"].notna()
        metrics["n_btts"] = int(m.sum())
        metrics["n_correct_btts"] = int(resolved.loc[m, "btts_correcte"].sum())

    if "corners_over_9_5_correcte" in resolved.columns:
        m = resolved["corners_over_9_5_correcte"].notna()
        metrics["n_corners_9_5"] = int(m.sum())
        metrics["n_correct_corners_9_5"] = int(resolved.loc[m, "corners_over_9_5_correcte"].sum())

    return metrics


def upsert_weekly_summary(week_key, metrics):
    row = {"semaine": week_key, **metrics}
    if WEEKLY_SUMMARY_PATH.exists():
        df = pd.read_csv(WEEKLY_SUMMARY_PATH)
        df = df[df["semaine"] != week_key]
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("semaine").reset_index(drop=True)
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WEEKLY_SUMMARY_PATH, index=False)
    print(f"weekly_summary.csv mis à jour pour la semaine {week_key}.")


# ------------------------------------------------------------------
# 5. CLI
# ------------------------------------------------------------------

def run_update_all():
    files = sorted(TRACKING_DIR.glob("predictions_*.csv"))
    if not files:
        print("Aucun fichier data/tracking/predictions_*.csv trouvé.")
        return

    for path in files:
        df = run_update_file(path)
        week_match = PREDICTIONS_FILENAME_RE.search(path.name)
        week_key = week_match.group(1) if week_match else path.stem
        metrics = compute_week_metrics(df)
        if metrics["n_matchs_resolus"] > 0:
            upsert_weekly_summary(week_key, metrics)


def _fmt_rate(n_correct, n_total):
    # bookmaker_correcte/*_correcte sont en dtype "object" (mélange de
    # True/False/NaN après un round-trip CSV) -> .sum() sur ce dtype peut
    # renvoyer le dernier bool brut (ex: False) au lieu d'un compte entier
    # quand il n'y a qu'une seule valeur -> cast explicite en int partout.
    n_correct, n_total = int(n_correct), int(n_total)
    if n_total == 0:
        return "—"
    return f"{n_correct}/{n_total} ({n_correct / n_total * 100:.0f}%)"


def _count_true(series):
    """Nombre de True dans une colonne *_correcte (dtype object, NaN mêlés)."""
    return int(series.fillna(False).astype(bool).sum())


def run_daily_report():
    """
    Affiche, jour par jour, le nombre de pronostics corrects sur le nombre
    total (pas juste une métrique abstraite) -> modèle ET bookmaker pour le
    1N2 (les deux jouent sur les mêmes matchs, comparables) ; modèle seul
    pour Buts/Corners (pas de cotes bookmaker suivies pour ces marchés ici).
    """
    files = sorted(TRACKING_DIR.glob("predictions_*.csv"))
    if not files:
        print("Aucun fichier data/tracking/predictions_*.csv trouvé.")
        return

    frames = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    resolved = df[df["resultat_reel"].notna()].copy()
    if resolved.empty:
        print("Aucun match résolu pour l'instant.")
        return

    print(f"{'Date':<12} {'1N2 modèle':<14} {'1N2 bookmaker':<15} {'Over 2,5':<12} {'BTTS':<12} {'Corners 9,5':<12}")
    for d, day_df in resolved.groupby(resolved["date"].dt.strftime("%Y-%m-%d")):
        modele = _fmt_rate((day_df["prediction_modele"] == day_df["resultat_reel"]).sum(), len(day_df))
        with_odds = day_df["bookmaker_correcte"].notna() if "bookmaker_correcte" in day_df else pd.Series(dtype=bool)
        bookmaker = _fmt_rate(day_df.get("bookmaker_correcte", pd.Series(dtype=bool)).sum(), with_odds.sum())
        og = day_df.get("over_2_5_correcte", pd.Series(dtype=bool))
        over25 = _fmt_rate(og.sum(), og.notna().sum())
        bt = day_df.get("btts_correcte", pd.Series(dtype=bool))
        btts = _fmt_rate(bt.sum(), bt.notna().sum())
        co = day_df.get("corners_over_9_5_correcte", pd.Series(dtype=bool))
        corners = _fmt_rate(co.sum(), co.notna().sum())
        print(f"{d:<12} {modele:<14} {bookmaker:<15} {over25:<12} {btts:<12} {corners:<12}")

    n = len(resolved)
    n_mod = int((resolved["prediction_modele"] == resolved["resultat_reel"]).sum())
    print()
    print(f"TOTAL : modèle {_fmt_rate(n_mod, n)} de bons pronostics 1N2 sur toute la période suivie.")
    if "bookmaker_correcte" in resolved.columns:
        m = resolved["bookmaker_correcte"].notna()
        print(f"        bookmaker {_fmt_rate(resolved.loc[m, 'bookmaker_correcte'].sum(), m.sum())} (sur les matchs où une cote était connue).")


def main():
    parser = argparse.ArgumentParser(description="Suivi hebdomadaire de la performance du modèle.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", action="store_true", help="Génère les prédictions des 7 prochains jours.")
    group.add_argument("--refresh-odds", action="store_true", help="Complète les cotes pas encore postées lors du --predict.")
    group.add_argument("--update", action="store_true", help="Complète les résultats réels + recalcule les métriques.")
    group.add_argument("--daily-report", action="store_true", help="Affiche le détail jour par jour (modèle vs bookmaker).")
    args = parser.parse_args()

    if args.predict:
        run_predict(date.today(), date.today() + timedelta(days=7))
    elif args.refresh_odds:
        run_refresh_odds_all()
    elif args.update:
        run_update_all()
    elif args.daily_report:
        run_daily_report()


if __name__ == "__main__":
    main()
