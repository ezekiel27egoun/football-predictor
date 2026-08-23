"""
Dérive les probabilités de marchés "cartons" à partir des cartons attendus
(lambda_home_yellow, lambda_away_yellow, lambda_red_total), via une loi de
Poisson -> même principe que goals_markets.py/corners_markets.py.

Jaunes : seuils par équipe + total du match, calés sur les moyennes
observées dans l'historique (~1,9 jaune/match pour le domicile, ~2,1 pour
l'extérieur, ~4,1 au total).

Rouges : UNE SEULE probabilité ("au moins un rouge dans le match"), pas de
seuil par équipe -> l'événement est trop rare (15,7% des matchs) pour un
découpage par équipe fiable, cf. décision explicite avec l'utilisateur.
"""
import numpy as np
from scipy.stats import poisson


def compute_cards_markets(lambda_home_yellow, lambda_away_yellow, lambda_red_total):
    lambda_home_yellow = np.maximum(lambda_home_yellow, 0.01)
    lambda_away_yellow = np.maximum(lambda_away_yellow, 0.01)
    lambda_red_total = np.maximum(lambda_red_total, 0.001)
    lambda_yellow_total = lambda_home_yellow + lambda_away_yellow

    return {
        "expected_home_yellow_cards": lambda_home_yellow,
        "expected_away_yellow_cards": lambda_away_yellow,
        "proba_home_yellow_over_1_5": 1 - poisson.cdf(1, lambda_home_yellow),
        "proba_home_yellow_over_2_5": 1 - poisson.cdf(2, lambda_home_yellow),
        "proba_away_yellow_over_1_5": 1 - poisson.cdf(1, lambda_away_yellow),
        "proba_away_yellow_over_2_5": 1 - poisson.cdf(2, lambda_away_yellow),
        "proba_yellow_over_3_5": 1 - poisson.cdf(3, lambda_yellow_total),
        "proba_yellow_over_4_5": 1 - poisson.cdf(4, lambda_yellow_total),
        "expected_red_cards": lambda_red_total,
        # P(au moins 1) = 1 - P(0) ; poisson.cdf(0, l) == P(exactement 0).
        "proba_red_card_in_match": 1 - poisson.cdf(0, lambda_red_total),
    }
