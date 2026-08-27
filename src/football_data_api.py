"""
Client minimal pour l'API football-data.org (plan gratuit).
Documentation : https://docs.football-data.org/
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.football-data.org/v4"

# Les 6 compétitions du projet, avec leur code football-data.org
COMPETITIONS = {
    "premier_league": "PL",
    "ligue_1": "FL1",
    "liga": "PD",
    "bundesliga": "BL1",
    "serie_a": "SA",
    "champions_league": "CL",
}


def _get_headers():
    # .strip() : un secret copié-collé (ex: GitHub Actions, .env) embarque
    # parfois un retour à la ligne en trop -> un header HTTP ne peut pas
    # contenir de \n, ce qui fait planter la requête avec une erreur peu
    # explicite ("Invalid header value") plutôt qu'un message clair.
    token = os.environ.get("FOOTBALL_DATA_API_TOKEN", "").strip() or None
    if not token:
        raise RuntimeError("FOOTBALL_DATA_API_TOKEN manquant : vérifie le fichier .env")
    return {"X-Auth-Token": token}


def _get_with_retry(url, params=None, max_retries=3, max_connection_retries=3):
    """
    GET avec deux types de nouvelle tentative automatique :
    - 429 (quota API dépassé, 10 requêtes/minute en plan gratuit) -> attend
      le délai indiqué par l'API (en-tête Retry-After), ou 60s par défaut.
    - erreur réseau/SSL transitoire (ex: SSLEOFError observé le 24/08,
      2 runs automatiques d'affilée en échec avant que ça reparte tout seul
      4h plus tard) -> avant, ces erreurs faisaient planter tout le run
      immédiatement, sans aucune tentative. Backoff court (5s, 15s, 30s),
      car une coupure réseau se résout généralement en quelques secondes,
      pas en minutes comme un quota dépassé.
    """
    connection_attempt = 0
    quota_attempt = 0
    while True:
        try:
            resp = requests.get(url, headers=_get_headers(), params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            connection_attempt += 1
            if connection_attempt > max_connection_retries:
                raise
            wait = [5, 15, 30][connection_attempt - 1]
            print(f"Erreur réseau ({e.__class__.__name__}) -> nouvelle tentative dans {wait}s "
                  f"({connection_attempt}/{max_connection_retries})...")
            time.sleep(wait)
            continue

        if resp.status_code != 429:
            # raise_for_status() seul ne montre que le code HTTP (ex: "400
            # Client Error") -> le corps de la réponse contient le vrai
            # message de l'API (souvent bien plus parlant), à afficher
            # explicitement avant de laisser l'erreur remonter.
            if not resp.ok:
                print(f"Réponse API ({resp.status_code}) : {resp.text[:500]}")
            resp.raise_for_status()
            return resp

        quota_attempt += 1
        if quota_attempt > max_retries:
            print(f"Réponse API ({resp.status_code}) : {resp.text[:500]}")
            resp.raise_for_status()  # dernière tentative -> on laisse l'erreur remonter
        wait = int(resp.headers.get("Retry-After", 60))
        print(f"Quota API dépassé (429), nouvelle tentative dans {wait}s...")
        time.sleep(wait)


def get_teams(competition_code):
    """Liste des équipes d'une compétition (nom officiel API + nom court)."""
    resp = _get_with_retry(f"{BASE_URL}/competitions/{competition_code}/teams")
    return resp.json()["teams"]


def get_matches(competition_code, status=None, date_from=None, date_to=None):
    """
    Récupère les matchs d'une compétition.
    status : "SCHEDULED", "FINISHED", "LIVE", etc. (None = tous statuts)
    date_from / date_to : format "YYYY-MM-DD"
    """
    params = {}
    if status:
        params["status"] = status
    if date_from:
        params["dateFrom"] = date_from
    if date_to:
        params["dateTo"] = date_to

    resp = _get_with_retry(f"{BASE_URL}/competitions/{competition_code}/matches", params=params)
    return resp.json()["matches"]


def get_all_teams(sleep_seconds=6):
    """
    Récupère les équipes des 6 compétitions du projet.
    sleep_seconds : pause entre les appels pour respecter le quota gratuit
    (10 requêtes/minute -> 6 secondes d'écart minimum).
    """
    all_teams = {}
    for league_name, code in COMPETITIONS.items():
        teams = get_teams(code)
        all_teams[league_name] = [
            {"api_name": t["name"], "api_short_name": t["shortName"], "api_tla": t["tla"]}
            for t in teams
        ]
        print(f"{league_name} ({code}) : {len(teams)} équipes")
        time.sleep(sleep_seconds)
    return all_teams
