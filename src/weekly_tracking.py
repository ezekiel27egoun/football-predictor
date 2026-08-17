"""
Suivi hebdomadaire de la performance du modèle : compare les probabilités de
rf_v8 aux résultats réels et aux cotes de bookmaker (The Odds API), semaine
après semaine.

Outil de MESURE uniquement — ne modifie ni ne ré-entraîne le modèle. Sert à
décider ensuite (ailleurs, à la main) s'il faut ré-entraîner plus souvent,
ajouter des features, calibrer les probabilités, etc.

Usage :
    python weekly_tracking.py --predict            # avant la journée : génère
                                                     # data/tracking/predictions_{date}.csv
    python weekly_tracking.py --refresh-odds        # en milieu de semaine : complète les
                                                     # cotes pas encore postées au moment du --predict
    python weekly_tracking.py --update              # après la journée : complète
                                                     # les résultats réels + métriques
                                                     # de tous les fichiers en attente
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
from fetch_fixtures import get_upcoming_fixtures
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
    key = os.environ.get("ODDS_API_KEY")
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


def run_update_file(path):
    df = pd.read_csv(path, parse_dates=["date"])
    # Tant qu'aucun résultat n'est connu, resultat_reel/prediction_correcte
    # sont 100% NaN -> pandas les lit en float64. On les repasse en object
    # avant d'y écrire des lettres ('H'/'D'/'A') ou des booléens.
    df["resultat_reel"] = df["resultat_reel"].astype("object")
    df["prediction_correcte"] = df["prediction_correcte"].astype("object")
    pending = df["resultat_reel"].isna()
    if not pending.any():
        print(f"{path.name} : déjà complet, rien à faire.")
        return df

    date_min, date_max = df.loc[pending, "date"].min(), df.loc[pending, "date"].max()
    real_results = fetch_real_results(date_min.date(), date_max.date())

    if real_results.empty:
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
    df.loc[newly_resolved, "resultat_reel"] = df.loc[newly_resolved].apply(
        lambda r: result_letter(r["home_score"], r["away_score"]), axis=1
    )
    df.loc[newly_resolved, "prediction_correcte"] = (
        df.loc[newly_resolved, "resultat_reel"] == df.loc[newly_resolved, "prediction_modele"]
    )
    df = df.drop(columns=["home_score", "away_score"])

    df.to_csv(path, index=False)
    n_new = int(newly_resolved.sum())
    print(f"{path.name} : {n_new} résultat(s) mis à jour ({int(pending.sum()) - n_new} encore en attente).")
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
    }
    if resolved.empty:
        return metrics

    metrics["accuracy_modele"] = float((resolved["prediction_modele"] == resolved["resultat_reel"]).mean())

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


def main():
    parser = argparse.ArgumentParser(description="Suivi hebdomadaire de la performance du modèle.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", action="store_true", help="Génère les prédictions des 7 prochains jours.")
    group.add_argument("--refresh-odds", action="store_true", help="Complète les cotes pas encore postées lors du --predict.")
    group.add_argument("--update", action="store_true", help="Complète les résultats réels + recalcule les métriques.")
    args = parser.parse_args()

    if args.predict:
        run_predict(date.today(), date.today() + timedelta(days=7))
    elif args.refresh_odds:
        run_refresh_odds_all()
    elif args.update:
        run_update_all()


if __name__ == "__main__":
    main()
