"""
Rafraîchit l'historique de matchs des 5 championnats domestiques avec les
résultats de la saison en cours, en retéléchargeant le CSV à jour depuis
football-data.co.uk (mis à jour régulièrement par le site lui-même tant que
la saison est en cours), puis reconstruit data/processed/matches_all_raw.csv.

La Champions League n'est PAS gérée ici : sa source (fbref, fichiers .xls)
n'a pas d'URL directe stable -> à mettre à jour manuellement en retéléchargeant
un fichier ucl_<saison>.xls dans data/raw/champions_league/ si besoin.

Usage :
    cd src
    python3 update_historical_data.py
"""
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from build_historical_dataset import build_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# code division football-data.co.uk -> (dossier local, préfixe de fichier)
LEAGUE_CODES = {
    "E0": ("premier_league", "premier_league"),
    "F1": ("ligue_1", "ligue_1"),
    "SP1": ("liga", "liga"),
    "D1": ("bundesliga", "bundesliga"),
    "I1": ("serie_a", "serie_a"),
}


def current_season_code(today=None):
    """
    Saison européenne = août année N -> mai année N+1.
    Retourne (code_url "2627", label "2026-2027") pour la saison en cours.
    """
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    end_year = start_year + 1
    code = f"{str(start_year)[-2:]}{str(end_year)[-2:]}"
    label = f"{start_year}-{end_year}"
    return code, label


def download_league(div_code, season_code):
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{div_code}.csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def update_all_leagues():
    season_code, season_label = current_season_code()
    print(f"Saison ciblée : {season_label} (code URL {season_code})\n")

    for div_code, (folder_name, file_prefix) in LEAGUE_CODES.items():
        try:
            csv_text = download_league(div_code, season_code)
        except requests.RequestException as e:
            print(f"[{folder_name}] Échec du téléchargement : {e}")
            continue

        # Le site redirige parfois vers un autre fichier (ou une page d'erreur)
        # quand le fichier de la saison n'existe pas encore -> on vérifie que
        # la colonne "Div" du fichier reçu correspond bien à ce qu'on a demandé.
        first_line = csv_text.splitlines()[1] if len(csv_text.splitlines()) > 1 else ""
        actual_div = first_line.split(",")[0].strip()
        if actual_div != div_code:
            print(f"[{folder_name}] Pas encore disponible pour {season_label} "
                  f"(reçu '{actual_div}' au lieu de '{div_code}') -> ignoré")
            continue

        dest = RAW_DIR / folder_name / f"{file_prefix}_{season_label}.csv"
        dest.write_text(csv_text, encoding="latin1", errors="replace")
        n_matches = csv_text.count("\n")
        print(f"[{folder_name}] OK -> {dest.name} (~{n_matches} lignes)")

    print("\nReconstruction de data/processed/matches_all_raw.csv...")
    build_dataset()


if __name__ == "__main__":
    update_all_leagues()
