"""
Dérive les probabilités de marchés "buts" (over/under, BTTS) à partir des
buts attendus par équipe (lambda_home, lambda_away), via une loi de Poisson
-> un seul calcul mathématique, pas un modèle par marché (cf. discussion :
cohérence garantie entre marchés, plutôt que N classifieurs indépendants
qui pourraient se contredire).

Hypothèse simplificatrice : buts domicile et extérieur indépendants (pas de
corrélation type Dixon-Coles) -> raisonnable en première approche, à
challenger plus tard si la calibration mesurée (weekly_tracking) le
justifie.
"""
import numpy as np
from scipy.stats import poisson


def compute_goals_markets(lambda_home, lambda_away):
    """
    lambda_home / lambda_away : buts attendus (float ou array-like), > 0.
    Retourne un dict de probabilités (ou de Series si l'entrée est une Series).
    """
    lambda_home = np.maximum(lambda_home, 0.01)  # Poisson mal défini à 0 exact
    lambda_away = np.maximum(lambda_away, 0.01)
    lambda_total = lambda_home + lambda_away

    return {
        "expected_home_goals": lambda_home,
        "expected_away_goals": lambda_away,
        "proba_over_1_5": 1 - poisson.cdf(1, lambda_total),
        "proba_over_2_5": 1 - poisson.cdf(2, lambda_total),
        "proba_over_3_5": 1 - poisson.cdf(3, lambda_total),
        "proba_btts_yes": (1 - poisson.pmf(0, lambda_home)) * (1 - poisson.pmf(0, lambda_away)),
    }
