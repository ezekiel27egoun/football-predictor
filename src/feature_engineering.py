"""
Feature engineering — extrait de notebooks/01_exploration.ipynb pour être
réutilisable à la fois par le notebook (entraînement) et par le pipeline de
prédiction (inférence sur des matchs à venir). Logique strictement identique
des deux côtés : toute divergence entre entraînement et inférence fausserait
silencieusement les prédictions.
"""
import numpy as np
import pandas as pd

STAT_COLS = [
    "goals_for", "goals_against", "shots_for", "shots_against",
    "shots_on_target_for", "shots_on_target_against",
    "corners_for", "corners_against", "fouls_for", "fouls_against", "points",
]

DOMESTIC_LEAGUES = ["premier_league", "ligue_1", "liga", "bundesliga", "serie_a"]

# Position par défaut pour une équipe absente d'une ligue sur les N dernières
# saisons connues (probable promotion récente) -> volontairement basse plutôt
# que la moyenne de ligue, pour ne pas la faire paraître "moyenne".
NEWLY_PROMOTED_DEFAULT_POSITION = 18.0


def build_team_perspective(df):
    """Transforme la base 1-ligne-par-match en 1-ligne-par-équipe-par-match."""
    home = pd.DataFrame({
        "date": df["date"],
        "league": df["league"],
        "season": df["season"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "is_home": True,
        "goals_for": df["full_time_home_goals"],
        "goals_against": df["full_time_away_goals"],
        "shots_for": df["home_shots"],
        "shots_against": df["away_shots"],
        "shots_on_target_for": df["home_shots_on_target"],
        "shots_on_target_against": df["away_shots_on_target"],
        "corners_for": df["home_corners"],
        "corners_against": df["away_corners"],
        "fouls_for": df["home_fouls"],
        "fouls_against": df["away_fouls"],
    })
    away = pd.DataFrame({
        "date": df["date"],
        "league": df["league"],
        "season": df["season"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "is_home": False,
        "goals_for": df["full_time_away_goals"],
        "goals_against": df["full_time_home_goals"],
        "shots_for": df["away_shots"],
        "shots_against": df["home_shots"],
        "shots_on_target_for": df["away_shots_on_target"],
        "shots_on_target_against": df["home_shots_on_target"],
        "corners_for": df["away_corners"],
        "corners_against": df["home_corners"],
        "fouls_for": df["away_fouls"],
        "fouls_against": df["home_fouls"],
    })

    team_matches = pd.concat([home, away], ignore_index=True)

    conditions = [
        team_matches["goals_for"] > team_matches["goals_against"],
        team_matches["goals_for"] == team_matches["goals_against"],
    ]
    team_matches["points"] = np.select(conditions, [3, 1], default=0)

    team_matches = team_matches.sort_values(["team", "date"]).reset_index(drop=True)
    return team_matches


def add_rolling_features(df, group_col="team", windows=(5, 10)):
    """Moyennes glissantes par équipe, décalées d'un cran (shift(1))."""
    df = df.sort_values([group_col, "date"]).copy()
    grouped = df.groupby(group_col, group_keys=False)

    for window in windows:
        for col in STAT_COLS:
            new_col = f"{col}_avg_last{window}"
            df[new_col] = grouped[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    df["matches_played_before"] = grouped.cumcount()
    return df


def add_rolling_features_by_venue(df, windows=(5, 10)):
    """Comme add_rolling_features, séparément pour domicile et extérieur."""
    df = df.sort_values(["team", "is_home", "date"]).copy()
    grouped = df.groupby(["team", "is_home"], group_keys=False)

    for window in windows:
        for col in STAT_COLS:
            new_col = f"{col}_avg_last{window}_venue"
            df[new_col] = grouped[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
    return df


def add_rest_days(df):
    """Jours depuis le match précédent de l'équipe, toutes compétitions confondues."""
    df = df.sort_values(["team", "date"]).copy()
    df["days_since_last_match"] = df.groupby("team")["date"].diff().dt.days
    return df


def add_head_to_head_features(df):
    """Historique des confrontations directes (moyenne expanding, shift(1))."""
    df = df.sort_values(["team", "opponent", "date"]).copy()
    grouped = df.groupby(["team", "opponent"], group_keys=False)

    df["goal_diff"] = df["goals_for"] - df["goals_against"]
    df["h2h_points_avg"] = grouped["points"].transform(lambda s: s.shift(1).expanding().mean())
    df["h2h_goal_diff_avg"] = grouped["goal_diff"].transform(lambda s: s.shift(1).expanding().mean())
    df["h2h_matches_played"] = grouped.cumcount()
    return df


def add_current_standings(df_team_matches):
    """
    Position au classement + points cumulés dans la saison EN COURS, avant
    le match du jour (shift(1)). Approximation nécessaire : les données
    historiques n'ont pas de numéro de journée synchronisé (certaines
    équipes ont joué un match de plus/moins à une date donnée) -> on classe
    les équipes par points-par-match joué (pas points bruts), ce qui limite
    ce biais, puis on reconstruit un "classement du jour" en propageant
    (ffill) le dernier points-par-match connu de chaque équipe sur toutes
    les dates de la saison.
    """
    df = df_team_matches.sort_values(["league", "season", "team", "date"]).copy()
    grouped = df.groupby(["league", "season", "team"], group_keys=False)

    points_before = grouped["points"].transform(lambda s: s.shift(1).cumsum())
    matches_before = grouped.cumcount()
    # Assignées ICI, tant que df n'a pas encore été passé dans un merge()
    # (qui repartirait d'un index 0..N-1 frais) -> sinon l'affectation par
    # Series se fait par alignement de LABEL d'index, pas par position, et
    # désaligne silencieusement les valeurs après un merge (bug constaté :
    # season_points_before divisé par ~2 pour Arsenal en fin de saison).
    df["ppg_before"] = points_before / matches_before.replace(0, np.nan)
    df["season_points_before"] = points_before.fillna(0)

    position_frames = []
    for (league, season), g in df.groupby(["league", "season"]):
        pivot = g.pivot_table(index="date", columns="team", values="ppg_before", aggfunc="last")
        pivot = pivot.sort_index().ffill()
        ranks = pivot.rank(axis=1, ascending=False, method="min")
        long_rank = ranks.stack().rename("league_position").reset_index()
        long_rank["league"] = league
        long_rank["season"] = season
        position_frames.append(long_rank)

    positions = pd.concat(position_frames, ignore_index=True)
    df = df.merge(positions, on=["league", "season", "team", "date"], how="left")
    return df


def add_domestic_position_lookup(df_team_matches):
    """
    Rattache à CHAQUE ligne (y compris Champions League) la position de
    l'équipe dans SON championnat domestique à cette date (la plus récente
    connue, via merge_asof) -> les matchs de C1 bénéficient aussi du niveau
    domestique de chaque équipe, pas seulement de son classement C1 interne.
    """
    domestic = (
        df_team_matches[df_team_matches["league"].isin(DOMESTIC_LEAGUES)]
        [["team", "date", "league_position"]]
        .dropna(subset=["league_position"])
        .rename(columns={"league_position": "domestic_position"})
        .sort_values("date")
    )

    df = df_team_matches.sort_values("date").copy()
    df = pd.merge_asof(
        df, domestic, on="date", by="team", direction="backward", allow_exact_matches=False
    )
    return df


def compute_final_standings(df_team_matches):
    """Classement final (points totaux) de chaque équipe pour chaque (league, season)."""
    totals = df_team_matches.groupby(["league", "season", "team"])["points"].sum().reset_index()
    totals["final_position"] = totals.groupby(["league", "season"])["points"].rank(
        ascending=False, method="min"
    )
    return totals[["league", "season", "team", "final_position"]]


def add_multi_season_strength(df_team_matches, n_seasons=3):
    """
    Position moyenne sur les 1 à `n_seasons` dernières saisons PRÉCÉDENTES
    (même ligue) -> capture la domination structurelle d'un club plutôt
    qu'une bonne/mauvaise saison isolée.
    - Équipe absente de cette ligue sur cette période alors qu'on a des
      données pour d'autres équipes à ce moment -> probable promotion
      récente -> NEWLY_PROMOTED_DEFAULT_POSITION (pas la moyenne de ligue).
    - Toute première saison de notre historique (aucune saison antérieure
      disponible pour QUI QUE CE SOIT) -> NaN, on ne peut rien en déduire,
      laissé au repli générique (moyenne de ligue) utilisé ailleurs.
    """
    final_standings = compute_final_standings(df_team_matches)
    seasons_sorted = sorted(final_standings["season"].unique())  # "YYYY-YYYY" -> tri lexicographique correct
    season_index = {s: i for i, s in enumerate(seasons_sorted)}
    lookup = final_standings.set_index(["league", "season", "team"])["final_position"]

    def previous_avg(row):
        idx = season_index.get(row["season"])
        if idx is None or idx == 0:
            return np.nan
        prev_seasons = seasons_sorted[max(0, idx - n_seasons):idx]
        vals = [lookup.get((row["league"], s, row["team"])) for s in prev_seasons]
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        if vals:
            return float(np.mean(vals))
        return NEWLY_PROMOTED_DEFAULT_POSITION

    df_team_matches = df_team_matches.copy()
    df_team_matches["prev_seasons_avg_position"] = df_team_matches.apply(previous_avg, axis=1)
    return df_team_matches


def compute_cl_champions(df_all):
    """
    Vainqueur de la Champions League pour chaque saison = équipe gagnante du
    dernier match joué chronologiquement cette saison-là (la finale est
    toujours le dernier match de la saison -> pas besoin de colonne
    "stage", absente de nos données sauvegardées).
    """
    cl = df_all[df_all["league"] == "champions_league"].dropna(
        subset=["full_time_home_goals", "full_time_away_goals"]
    )
    if cl.empty:
        return {}
    finals = cl.loc[cl.groupby("season")["date"].idxmax()].copy()
    finals["champion"] = np.where(
        finals["full_time_home_goals"] > finals["full_time_away_goals"],
        finals["home_team"], finals["away_team"],
    )
    return finals.set_index("season")["champion"].to_dict()


def add_reigning_champion(df_all, df_team_matches):
    """Ajoute reigning_champion (0/1) : l'équipe a-t-elle gagné la Champions League la saison précédente ?"""
    champions = compute_cl_champions(df_all)
    seasons_sorted = sorted(df_team_matches["season"].unique())
    season_index = {s: i for i, s in enumerate(seasons_sorted)}

    def is_reigning_champion(row):
        idx = season_index.get(row["season"])
        if idx is None or idx == 0:
            return 0
        previous_season = seasons_sorted[idx - 1]
        return int(champions.get(previous_season) == row["team"])

    df_team_matches = df_team_matches.copy()
    df_team_matches["reigning_champion"] = df_team_matches.apply(is_reigning_champion, axis=1)
    return df_team_matches


def compute_elo_ratings(df, k=20, initial_rating=1500, home_advantage=100):
    """Rating Elo par équipe, mis à jour séquentiellement match après match."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    ratings = {}

    home_elo_before = []
    away_elo_before = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_rating = ratings.get(home, initial_rating)
        away_rating = ratings.get(away, initial_rating)

        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)

        expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))

        # Résultat inconnu (match futur) -> pas de mise à jour, on garde le rating tel quel
        if pd.isna(row["full_time_home_goals"]) or pd.isna(row["full_time_away_goals"]):
            ratings[home] = home_rating
            ratings[away] = away_rating
            continue

        if row["full_time_home_goals"] > row["full_time_away_goals"]:
            actual_home = 1.0
        elif row["full_time_home_goals"] == row["full_time_away_goals"]:
            actual_home = 0.5
        else:
            actual_home = 0.0

        ratings[home] = home_rating + k * (actual_home - expected_home)
        ratings[away] = away_rating + k * ((1 - actual_home) - (1 - expected_home))

    df["home_elo_before"] = home_elo_before
    df["away_elo_before"] = away_elo_before
    return df


def build_features(df_all):
    """
    Pipeline complet : df_all (1 ligne/match) -> df_features (1 ligne/match
    + toutes les colonnes de forme). Fonctionne aussi bien pour des matchs
    déjà joués (entraînement) que pour des lignes "fantômes" de matchs
    futurs (full_time_home_goals/away_goals = NaN) utilisées en inférence.
    """
    df_all = compute_elo_ratings(df_all)

    df_team_matches = build_team_perspective(df_all)
    df_team_matches = add_rolling_features(df_team_matches)
    df_team_matches = add_rolling_features_by_venue(df_team_matches)
    df_team_matches = add_rest_days(df_team_matches)
    df_team_matches = add_head_to_head_features(df_team_matches)
    df_team_matches = add_current_standings(df_team_matches)
    df_team_matches = add_domestic_position_lookup(df_team_matches)
    df_team_matches = add_multi_season_strength(df_team_matches)
    df_team_matches = add_reigning_champion(df_all, df_team_matches)

    feature_cols = [c for c in df_team_matches.columns if "_avg_last" in c] + [
        "matches_played_before", "days_since_last_match",
        "h2h_points_avg", "h2h_goal_diff_avg", "h2h_matches_played",
        "league_position", "season_points_before", "domestic_position",
        "prev_seasons_avg_position", "reigning_champion",
    ]

    home_features = df_team_matches.loc[df_team_matches["is_home"], ["date", "team"] + feature_cols].copy()
    home_features = home_features.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in feature_cols}})

    away_features = df_team_matches.loc[~df_team_matches["is_home"], ["date", "team"] + feature_cols].copy()
    away_features = away_features.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in feature_cols}})

    df_features = df_all.merge(home_features, on=["date", "home_team"], how="left")
    df_features = df_features.merge(away_features, on=["date", "away_team"], how="left")

    # Features d'écart domicile/extérieur (cf. 02_modeling.ipynb) : uniquement
    # pour les stats de forme glissante (_avg_last...) et matches_played_before
    # -> pas pour days_since_last_match, h2h_* ou elo_before, qui n'en avaient
    # jamais fait partie lors de l'entraînement.
    diff_base_cols = [
        c[len("home_"):] for c in df_features.columns
        if c.startswith("home_") and ("_avg_last" in c or c.endswith("matches_played_before"))
    ]
    for col in diff_base_cols:
        df_features[f"diff_{col}"] = df_features[f"home_{col}"] - df_features[f"away_{col}"]

    return df_features
