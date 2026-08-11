"""
Récupère les matchs à venir (fixtures) des 6 compétitions depuis
football-data.org, et applique la table de correspondance des noms
d'équipes pour les relier à notre historique.
"""
import time
from pathlib import Path

import pandas as pd

from football_data_api import COMPETITIONS, get_matches

TEAM_MAPPING_PATH = Path(__file__).resolve().parent.parent / "data" / "team_name_mapping.csv"


def load_team_mapping():
    """
    Retourne un dict {(league, api_name): historical_name}.
    historical_name == "" ou NaN -> équipe sans historique (nouvelle).
    """
    df = pd.read_csv(TEAM_MAPPING_PATH)
    mapping = {}
    for _, row in df.iterrows():
        historical_name = row["confirmed_historical_name"]
        if pd.isna(historical_name):
            historical_name = ""
        mapping[(row["league"], row["api_name"])] = historical_name
    return mapping


def get_upcoming_fixtures(date_from, date_to, sleep_seconds=6.5):
    """
    date_from / date_to : "YYYY-MM-DD"
    sleep_seconds : pause entre les 6 appels (une par compétition) pour
    respecter le quota gratuit (10 requêtes/minute).
    Retourne un DataFrame : date, league, season, home_team, away_team,
    home_team_known, away_team_known (bool -> False si équipe sans historique)
    """
    mapping = load_team_mapping()
    rows = []

    for i, (league_name, code) in enumerate(COMPETITIONS.items()):
        if i > 0:
            time.sleep(sleep_seconds)
        matches = get_matches(code, status="SCHEDULED", date_from=date_from, date_to=date_to)
        for m in matches:
            home_api_name = m["homeTeam"]["name"]
            away_api_name = m["awayTeam"]["name"]

            home_hist = mapping.get((league_name, home_api_name), None)
            away_hist = mapping.get((league_name, away_api_name), None)

            if home_hist is None or away_hist is None:
                # Equipe absente de la table de correspondance -> à investiguer
                # (nouvelle équipe jamais vue par build_team_mapping.py)
                print(f"ATTENTION : équipe non trouvée dans le mapping -> "
                      f"{home_api_name if home_hist is None else away_api_name} ({league_name})")

            # home_hist / away_hist : "" (mappé mais sans historique) ou None
            # (absent du mapping, ne devrait pas arriver) -> dans les deux cas
            # pas d'historique utilisable, on garde le nom API pour l'affichage
            rows.append({
                "date": m["utcDate"][:10],
                "league": league_name,
                "season": m["season"]["id"],  # id numérique côté API, pas comparable à nos labels "2025-2026"
                "home_team_api": home_api_name,
                "away_team_api": away_api_name,
                "home_crest": m["homeTeam"].get("crest", ""),
                "away_crest": m["awayTeam"].get("crest", ""),
                "home_team": home_hist if home_hist else home_api_name,
                "away_team": away_hist if away_hist else away_api_name,
                "home_team_known": bool(home_hist),
                "away_team_known": bool(away_hist),
                "matchday": m.get("matchday"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
