"""
Collecte automatique des résultats Champions League (fbref.com)
sur les 10 dernières saisons.

A LANCER EN LOCAL (VS Code / terminal Windows), pas dans un notebook cloud,
car fbref bloque les IP de datacenter.

Usage :
    python collect_champions_league.py

Résultat :
    Un fichier CSV par saison dans data/raw/champions_league/
"""

import pandas as pd
import requests
import time
import os

# ---- Configuration ----
SEASONS = ["2016-2017", "2017-2018", "2018-2019",
    "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025","2025-2026"]

OUTPUT_DIR = "data/raw/champions_league"
DELAY_SECONDS = 6  # important : fbref bloque si on va trop vite

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def collect_season(season: str) -> pd.DataFrame | None:
    """Télécharge et parse le tableau des résultats pour une saison donnée."""
    url = (
        f"https://fbref.com/en/comps/8/{season}/schedule/"
        f"{season}-Champions-League-Scores-and-Fixtures"
    )
    print(f"-> Téléchargement saison {season} ...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        print(f"   Erreur réseau pour {season} : {e}")
        return None

    if response.status_code != 200:
        print(f"   Echec (code {response.status_code}) pour {season}. "
              f"Site probablement bloqué ou URL invalide.")
        return None

    try:
        tables = pd.read_html(response.text)
    except ValueError:
        print(f"   Aucun tableau trouvé pour {season}.")
        return None

    # Le tableau des scores est généralement le premier de la page
    df = tables[0]
    df["season"] = season
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_seasons = []

    for season in SEASONS:
        df = collect_season(season)
        if df is not None:
            filename = os.path.join(OUTPUT_DIR, f"ucl_{season}.csv")
            df.to_csv(filename, index=False)
            print(f"   Sauvegardé : {filename} ({len(df)} lignes)")
            all_seasons.append(df)
        # Pause obligatoire entre deux requêtes pour ne pas se faire bloquer
        time.sleep(DELAY_SECONDS)

    if all_seasons:
        combined = pd.concat(all_seasons, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "ucl_all_seasons_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nFichier combiné créé : {combined_path} ({len(combined)} lignes)")
    else:
        print("\nAucune saison n'a pu être collectée. "
              "Voir la méthode manuelle de secours (Share & Export sur fbref).")


if __name__ == "__main__":
    main()
