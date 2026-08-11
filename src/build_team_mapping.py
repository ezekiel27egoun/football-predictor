"""
Construit une table de correspondance entre les noms d'équipes de l'API
football-data.org et les noms utilisés dans nos données historiques
(football-data.co.uk / fbref).

Sortie : data/team_name_mapping.csv, à relire et corriger à la main pour les
lignes à faible score de confiance avant de s'en servir dans le pipeline.
"""
import difflib
import re
import unicodedata

import pandas as pd

from football_data_api import COMPETITIONS, get_all_teams


def normalize(name):
    """Nettoie un nom d'équipe pour faciliter le rapprochement (accents, suffixes club)."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    # Suffixes/préfixes fréquents qui varient d'une source à l'autre
    tokens_to_drop = {"fc", "afc", "cf", "sc", "ac", "ssc", "ud", "cd", "calcio", "club", "1", "04", "05"}
    words = re.findall(r"[a-z0-9]+", name)
    words = [w for w in words if w not in tokens_to_drop]
    return " ".join(words)


def best_match(target, candidates):
    """Retourne (meilleur_candidat, score) parmi une liste, ou (None, 0) si rien d'assez proche."""
    if not candidates:
        return None, 0.0
    scores = [
        (c, difflib.SequenceMatcher(None, normalize(target), normalize(c)).ratio())
        for c in candidates
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0]


def main():
    df_hist = pd.read_csv("../data/processed/matches_all_raw.csv")

    print("Récupération des équipes depuis l'API (peut prendre ~40s, quota gratuit)...")
    api_teams = get_all_teams()

    rows = []
    for league_name in COMPETITIONS:
        historical_names = sorted(
            set(df_hist.loc[df_hist["league"] == league_name, "home_team"]).union(
                df_hist.loc[df_hist["league"] == league_name, "away_team"]
            )
        )
        for team in api_teams[league_name]:
            # On essaie le nom complet ET le nom court de l'API, on garde le meilleur des deux
            candidate_full, score_full = best_match(team["api_name"], historical_names)
            candidate_short, score_short = best_match(team["api_short_name"], historical_names)
            if score_short > score_full:
                proposed, score = candidate_short, score_short
            else:
                proposed, score = candidate_full, score_full

            rows.append({
                "league": league_name,
                "api_name": team["api_name"],
                "api_short_name": team["api_short_name"],
                "proposed_historical_name": proposed,
                "score": round(score, 3),
                "confirmed_historical_name": proposed if score >= 0.75 else "",
            })

    df_mapping = pd.DataFrame(rows).sort_values(["league", "score"])
    df_mapping.to_csv("../data/team_name_mapping.csv", index=False)

    n_total = len(df_mapping)
    n_low_confidence = (df_mapping["score"] < 0.75).sum()
    print(f"\n{n_total} équipes traitées, {n_low_confidence} à vérifier manuellement (score < 0.75)")
    print("-> data/team_name_mapping.csv")


if __name__ == "__main__":
    main()
