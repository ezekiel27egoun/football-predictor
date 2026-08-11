"""
Renomme automatiquement les fichiers CSV football-data.co.uk
en se basant sur la VRAIE date des matchs contenus dans chaque fichier,
plutot que sur l'ordre de telechargement (E0.csv, E0 (1).csv, etc.)

A lancer depuis le dossier qui contient les fichiers a renommer,
par exemple : data/raw/premier_league/

Usage :
    cd data/raw/premier_league
    python3 rename_league_files.py

Adapte la variable PREFIX pour chaque championnat :
    Premier League -> "premier_league"
    Ligue 1         -> "ligue_1"
    Liga            -> "liga"
    Bundesliga      -> "bundesliga"
    Serie A         -> "serie_a"
"""

import pandas as pd
import glob
import os

# ---- A adapter a chaque championnat ----
PREFIX = "bundesliga"   # <-- change ceci selon le dossier

def detect_season(filepath: str) -> str | None:
    """Lit un CSV football-data.co.uk et determine la saison a partir des dates."""
    try:
        df = pd.read_csv(filepath, encoding="latin1")
    except Exception as e:
        print(f"   Impossible de lire {filepath} : {e}")
        return None

    # La colonne de date s'appelle "Date" dans les fichiers football-data.co.uk
    if "Date" not in df.columns:
        print(f"   Pas de colonne 'Date' trouvee dans {filepath}")
        return None

    # Les dates peuvent etre en format JJ/MM/AAAA ou JJ/MM/AA selon les saisons
    dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    dates = dates.dropna()

    if dates.empty:
        print(f"   Aucune date valide trouvee dans {filepath}")
        return None

    min_date = dates.min()
    max_date = dates.max()

    # Une saison de football commence generalement en aout et finit en mai/juin
    # On prend l'annee de la date la plus ancienne comme annee de debut
    start_year = min_date.year
    end_year = max_date.year

    if start_year == end_year:
        # Cas rare : toutes les dates dans la meme annee civile
        season = f"{start_year}-{start_year + 1}"
    else:
        season = f"{start_year}-{end_year}"

    print(f"   Dates trouvees : {min_date.date()} -> {max_date.date()} "
          f"=> saison detectee : {season}")
    return season


def main():
    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("Aucun fichier CSV trouve dans ce dossier.")
        return

    print(f"{len(csv_files)} fichiers trouves.\n")

    renamed = []
    for filepath in csv_files:
        print(f"-> Analyse de {filepath}")
        season = detect_season(filepath)

        if season is None:
            print(f"   IGNORE : {filepath} n'a pas pu etre analyse.\n")
            continue

        new_name = f"{PREFIX}_{season}.csv"

        if os.path.exists(new_name) and new_name != filepath:
            print(f"   ATTENTION : {new_name} existe deja, renommage ignore "
                  f"pour eviter d'ecraser un fichier.\n")
            continue

        os.rename(filepath, new_name)
        print(f"   Renomme : {filepath} -> {new_name}\n")
        renamed.append(new_name)

    print(f"\nTermine. {len(renamed)} fichiers renommes avec succes.")


if __name__ == "__main__":
    main()
