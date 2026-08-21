"""
Dérive les probabilités de marchés "corners" (over/under) à partir des
corners attendus par équipe (lambda_home, lambda_away), via une loi de
Poisson -> même principe que goals_markets.py.

Seuils calés sur les moyennes observées dans l'historique (5 championnats
domestiques) : ~5,4 corners/match pour le domicile, ~4,4 pour l'extérieur,
~9,8 au total.
"""
import numpy as np
from scipy.stats import poisson


def compute_corners_markets(lambda_home, lambda_away):
    lambda_home = np.maximum(lambda_home, 0.01)
    lambda_away = np.maximum(lambda_away, 0.01)
    lambda_total = lambda_home + lambda_away

    return {
        "expected_home_corners": lambda_home,
        "expected_away_corners": lambda_away,
        "proba_home_corners_over_3_5": 1 - poisson.cdf(3, lambda_home),
        "proba_home_corners_over_4_5": 1 - poisson.cdf(4, lambda_home),
        "proba_away_corners_over_3_5": 1 - poisson.cdf(3, lambda_away),
        "proba_away_corners_over_4_5": 1 - poisson.cdf(4, lambda_away),
        "proba_corners_over_8_5": 1 - poisson.cdf(8, lambda_total),
        "proba_corners_over_9_5": 1 - poisson.cdf(9, lambda_total),
        "proba_corners_over_10_5": 1 - poisson.cdf(10, lambda_total),
    }
