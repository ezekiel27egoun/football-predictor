# Projet football-predictor

## Objectif
Modèle ML de prédiction de résultats de matchs de football (1N2 pour commencer),
sur les 5 grands championnats européens + Champions League, avec ambition
d'étendre vers d'autres marchés (buts, corners, cartons) une fois le premier
modèle validé. Inspiré de la logique déjà utilisée sur un projet ML de risque
budgétaire (Random Forest, scikit-learn).

Ambition à terme : repérer des écarts entre les probabilités du modèle et
celles impliquées par les cotes de bookmaker (value bets) — d'où la décision
de ne **jamais** utiliser les cotes comme feature d'entraînement (cf.
Décisions).

## Sources de données
- **5 championnats domestiques** (Premier League, Ligue 1, Liga, Bundesliga, Serie A) :
  CSV football-data.co.uk, 10 saisons (2016-2017 à 2025-2026), dans
  `data/raw/{league}/` (ex: `data/raw/premier_league/premier_league_2023-2024.csv`)
- **Champions League** : fichiers `.xls` exportés depuis fbref.com (Share & Export
  > Get table as CSV), en réalité du HTML — se chargent avec `pd.read_html()`,
  PAS `pd.read_excel()`. 10 saisons dans `data/raw/champions_league/ucl_{season}.xls`
  - Depuis 2024-2025 : nouveau format UEFA "ligue unique" à 36 équipes
    (197 matchs/saison au lieu de 131 avant) — colonne `Round` contient
    "League phase" au lieu de "Group stage"
  - Les scores avec tirs au but s'affichent `"1 (4)–1 (2)"` — extraction par regex
    pour ne garder que le score du temps réglementaire
  - Pas de stats détaillées dispo pour la Champions League (tirs, corners,
    cartons, cotes) — uniquement le score

## Pipeline notebooks/01_exploration.ipynb (chargement + feature engineering)
1. Chargement des 5 championnats en boucle, colonnes numériques séparées
   (`home_corners`, `away_corners`, etc., pas fusionnées en texte)
2. Chargement des 10 saisons Champions League avec nettoyage (codes pays,
   encodage, extraction buts par regex)
3. Harmonisation + fusion en `df_all` (19 309 matchs, 6 compétitions)
   — conversion dates avec `format="mixed", dayfirst=True` (les saisons
   anciennes des championnats ont une année à 2 chiffres, ex: `13/08/16`)
4. Sauvegarde : `data/processed/matches_all_raw.csv`
5. **Feature engineering** (toutes les features en `shift(1)` ou équivalent,
   jamais le match du jour — pas de fuite de données) :
   - Rolling stats 5/10 derniers matchs par équipe (buts, tirs, tirs cadrés,
     corners, fautes, points) — `home_*_avg_last5/10`, `away_*_avg_last5/10`
   - Même chose séparément pour domicile/extérieur (`groupby(["team","is_home"])`)
     — suffixe `_venue`
   - `days_since_last_match` : repos depuis le dernier match (fatigue,
     enchaînement championnat + Coupe d'Europe)
   - Head-to-head (`h2h_points_avg`, `h2h_goal_diff_avg`, `h2h_matches_played`) :
     moyenne **expanding** (pas de fenêtre fixe, confrontations trop rares)
   - Rating Elo (`home_elo_before`, `away_elo_before`) : calcul séquentiel
     (boucle `for`, pas vectorisable), k=20, home_advantage=100, jamais de NaN
   - Sauvegarde : `data/processed/matches_features.csv` (19 309 lignes, 124 colonnes)

## Pipeline notebooks/02_modeling.ipynb (modélisation)
1. Split chronologique train/validation/test (70/15/15, `shuffle=False`)
2. Ajout de features `diff_<stat> = home_<stat> - away_<stat>` (écart
   domicile/extérieur) — cf. leçon ci-dessous, n'a pas amélioré le score
   mais reste dans le pipeline
3. Cible recalculée depuis les buts (`result` H/D/A), pas depuis la colonne
   brute `full_time_result` (vide pour la Champions League)
4. `feature_cols` : uniquement les features de forme (`_avg_last`,
   `matches_played_before`, `days_since_last_match`, `h2h_*`, `elo_before`)
   — jamais les stats brutes du match du jour
5. Gestion des NaN : `dropna` pour la forme glissante (premiers matchs d'une
   équipe), `fillna(0)` spécifiquement pour les colonnes `h2h_points_avg`/
   `h2h_goal_diff_avg` (sinon on perdrait une part énorme du dataset — deux
   équipes précises se recroisent rarement)
6. 9 versions de modèle comparées sur validation (`class_weight="balanced"`,
   sélection par **f1-macro** et non accuracy — l'accuracy peut être
   maximisée en ignorant complètement les nuls)
7. Modèle retenu : **`rf_v8`** (Random Forest, `max_depth=6, min_samples_leaf=50,
   class_weight="balanced"`, 124 features cumulées)
8. Résultat honnête sur test (jamais retouché sauf une fois pour comparer à
   Gradient Boosting) : **accuracy 0.486, f1-macro 0.465** (baseline
   toujours-domicile ≈ 0.43)
9. Sauvegarde : `models/rf_v8.joblib` + `models/feature_cols.joblib`
   (liste exacte des colonnes attendues, dans l'ordre)

## Décisions prises
- Garder les colonnes numériques séparées plutôt que fusionnées en texte —
  nécessaire pour les moyennes glissantes.
- Split train/validation/test **chronologique**, jamais aléatoire.
- Rolling stats sur les N derniers matchs joués (pas une fenêtre de dates) —
  peut mélanger fin de saison précédente et championnat/Coupe d'Europe.
- **Ne jamais utiliser les cotes de bookmaker comme feature d'entraînement**
  — sinon le modèle apprend à reproduire le marché plutôt qu'à le challenger,
  ce qui viderait de son sens l'ambition value-bet du projet. Les cotes
  restent disponibles dans les données brutes pour servir de benchmark
  d'évaluation plus tard (comparer nos probabilités aux leurs), pas
  d'input.
- Sélection de modèle sur **f1-macro**, pas accuracy, à cause du déséquilibre
  de classes (les nuls sont sous-représentés et le modèle les ignore sinon).

## Leçons apprises (feature engineering, campagne v3→v9)
- **`feature_importances_` mesure "ce qui a été utilisé", pas "ce qui était
  indispensable"** : les features `diff_*` dominent le classement
  d'importance alors qu'elles n'ont mesurément rien changé à l'accuracy/f1 —
  un Random Forest peut déjà reconstruire une différence en combinant deux
  splits, donc leur ajout ne fait que changer le chemin de décision interne,
  pas la performance finale.
- **Rendements décroissants nets** : forme domicile/extérieur a apporté un
  vrai gain (+0.03 f1-macro), repos/head-to-head/Elo quasi rien
  individuellement (+0.001 à +0.002 chacun) — la plupart des features
  couvrent le même terrain d'information une fois la forme de base captée.
- **Random Forest vs Gradient Boosting : aucun gain net** sur ce dataset —
  confirme que le plafond (~accuracy 0.49, f1-macro 0.46 sur test) vient de
  l'information disponible dans les features, pas du choix d'algorithme.
- **Sélection de modèle répétée sur validation = léger biais optimiste** :
  8 comparaisons successives sur le même jeu de validation gonflent
  légèrement le score annoncé — d'où l'écart validation (0.507) → test
  (0.486) observé au moment de l'évaluation finale, attendu et pas alarmant
  vu son ampleur modeste.
- Un modèle à ~49% sur 3 classes n'est pas inutile : le baseline naïf est à
  ~43%, et le foot a une part d'aléa irréductible (même les bookmakers,
  avec bien plus de données, ne "devinent" pas le résultat exact — leur
  rentabilité vient d'une marge structurelle sur les cotes, pas d'une
  prédiction quasi parfaite). L'objectif réaliste est des **probabilités
  bien calibrées** (`predict_proba`), pas une prédiction dure toujours
  juste.

## Prochaine étape (pas encore commencée)
- Basculer vers des probabilités calibrées (`predict_proba`) plutôt qu'une
  prédiction dure H/D/A, et évaluer avec des métriques de calibration
  (log-loss, Brier score) plutôt qu'accuracy seule.
- Comparer ces probabilités aux cotes de bookmaker (implied probability =
  1/cote) pour repérer d'éventuels écarts — SANS jamais avoir entraîné le
  modèle sur les cotes elles-mêmes.
- Pistes de features encore inexplorées si besoin de dépasser le plateau
  actuel : qualité d'effectif/valeur marchande (si source de données
  trouvée), enjeu du match (course au titre, relégation, phase finale de
  coupe d'Europe).

## Environnement
- macOS, VS Code, Anaconda (`base` env, Python 3.13.9)
- Chemin projet : `~/Documents/PROJETS/"EGK SkillForge"/football-predictor`
  (attention à l'espace dans "EGK SkillForge", toujours entre guillemets dans le terminal)
- fbref.com bloque les requêtes automatisées (403, Cloudflare) même depuis IP
  résidentielle — collecte Champions League faite manuellement via
  Share & Export, pas de scraping automatique possible sur ce site
- Notebooks exécutés dans VS Code : après une modification faite par Claude
  (édition du fichier .ipynb sur disque), l'onglet ouvert ne se recharge pas
  automatiquement — faire **"Revert File"** (Cmd+Shift+P) pour voir les
  nouvelles cellules, puis **"Run Cell and Below"** à partir de la cellule
  modifiée plutôt que "Run All" (le kernel garde les variables déjà
  calculées en mémoire, pas besoin de tout rejouer).
