"""
App Streamlit : probabilités H/D/A pour les matchs à venir des 5 grands
championnats européens + Champions League.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict_matches import predict_upcoming_matches  # noqa: E402

LEAGUE_LABELS = {
    "premier_league": "Premier League",
    "ligue_1": "Ligue 1",
    "liga": "Liga",
    "bundesliga": "Bundesliga",
    "serie_a": "Serie A",
    "champions_league": "Champions League",
}

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")

# --- Palette (catégorielle, validée contraste + daltonisme, cf. skill dataviz) ---
STYLE = """
<style>
:root {
  --surface-1:      #fcfcfb;
  --page-plane:      #f9f9f7;
  --text-primary:    #0b0b0b;
  --text-secondary:  #52514e;
  --text-muted:      #898781;
  --border:          rgba(11,11,11,0.10);
  --home:            #2a78d6;
  --draw:            #eb6834;
  --away:            #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-fp-theme="light"]) {
    --surface-1:      #1a1a19;
    --page-plane:      #0d0d0d;
    --text-primary:    #ffffff;
    --text-secondary:  #c3c2b7;
    --border:          rgba(255,255,255,0.10);
    --home:            #3987e5;
    --draw:            #d95926;
    --away:            #199e70;
  }
}

.fp-legend { display:flex; gap:20px; align-items:center; margin: 4px 0 18px 0;
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-secondary); }
.fp-legend .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }

.fp-league-header { font: 600 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.04em;
  margin: 18px 0 8px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border); }

.fp-card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 16px; margin-bottom: 10px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }

.fp-teams { display:flex; align-items:center; justify-content:space-between; margin-bottom: 10px; }
.fp-team { display:flex; align-items:center; gap:8px; font-size: 15px; font-weight: 600; color: var(--text-primary); }
.fp-team img { width: 24px; height: 24px; object-fit: contain; }
.fp-vs { color: var(--text-muted); font-size: 12px; padding: 0 10px; }
.fp-new { font-size: 11px; color: var(--text-muted); font-weight: 400; margin-left: 4px; }

.fp-bar { display:flex; height: 16px; border-radius: 4px; overflow: hidden; background: var(--page-plane); }
.fp-bar div { height: 100%; }
.fp-bar .seg-h { background: var(--home); }
.fp-bar .seg-d { background: var(--draw); }
.fp-bar .seg-a { background: var(--away); }
.fp-bar .seg-h + div, .fp-bar .seg-d + div { margin-left: 2px; }

.fp-values { display:flex; justify-content:space-between; margin-top: 6px;
  font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-secondary);
  font-variant-numeric: tabular-nums; }
.fp-values span.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; vertical-align:0px; }

.fp-date-header { font: 700 20px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary); margin: 24px 0 4px 0; }

.fp-hero { margin: 4px 0 6px 0; }
.fp-hero h1 { font: 800 40px/1.15 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; letter-spacing: -0.01em;
  background: linear-gradient(90deg, var(--home) 0%, var(--draw) 55%, var(--away) 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.fp-byline { font: 500 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-secondary); margin: 2px 0 18px 0; }
.fp-byline b { color: var(--text-primary); }

.fp-disclaimer { display:flex; gap:10px; align-items:flex-start; background: var(--surface-1);
  border: 1px solid var(--border); border-left: 3px solid var(--draw); border-radius: 8px;
  padding: 10px 14px; margin: 0 0 22px 0;
  font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-secondary); }
.fp-disclaimer .icon { font-size: 16px; line-height: 1.5; }

.fp-kpis { display:flex; gap:12px; margin: 0 0 20px 0; flex-wrap: wrap; }
.fp-kpi { flex:1; min-width: 140px; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.fp-kpi .v { font: 700 24px/1.2 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-primary); }
.fp-kpi .l { font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-muted); }

.fp-pill { font: 600 11px/1; padding: 4px 8px; border-radius: 20px; white-space: nowrap; }
.fp-pill.strong { background: color-mix(in srgb, var(--home) 16%, transparent); color: var(--home); }
.fp-pill.mid { background: color-mix(in srgb, var(--draw) 16%, transparent); color: var(--draw); }
.fp-pill.open { background: var(--page-plane); color: var(--text-muted); border: 1px solid var(--border); }
.fp-teams-row { display:flex; align-items:center; justify-content:space-between; margin-bottom: 6px; }
.fp-team.favored { color: var(--favored-color, var(--text-primary)); }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # 1h de cache -> évite de re-solliciter l'API à chaque interaction
def get_predictions(date_from, date_to):
    return predict_upcoming_matches(str(date_from), str(date_to))


st.markdown(
    '<div class="fp-hero"><h1>⚽ Football Predictor</h1></div>'
    '<div class="fp-byline">Projet personnel de <b>Ezekiel Egounleti</b> — '
    'Random Forest entraîné sur l\'historique des 10 dernières saisons </div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="fp-disclaimer"><span class="icon">ℹ️</span>'
    '<span>Ces probabilités sont issues d\'un modèle statistique réalisé dans un cadre '
    "d'apprentissage personnel. Elles ne constituent en aucun cas un conseil ou une "
    'incitation au pari sportif.</span></div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([2, 2, 3])
with col1:
    date_from = st.date_input("Du", value=date.today())
with col2:
    date_to = st.date_input("Au", value=date.today() + timedelta(days=7))

with st.spinner("Récupération des matchs à venir et calcul des probabilités..."):
    df = get_predictions(date_from, date_to)

if df.empty:
    st.info("Aucun match programmé sur cette période pour les compétitions suivies.")
    st.stop()

with col3:
    available_leagues = [lg for lg in LEAGUE_LABELS if lg in df["league"].unique()]
    selected_leagues = st.multiselect(
        "Compétitions",
        options=available_leagues,
        default=available_leagues,
        format_func=lambda lg: LEAGUE_LABELS.get(lg, lg),
    )

df = df[df["league"].isin(selected_leagues)]
if df.empty:
    st.info("Aucun match pour les compétitions sélectionnées sur cette période.")
    st.stop()

search = st.text_input("🔎 Rechercher une équipe", placeholder="ex : Marseille, Real Madrid…")
if search:
    mask = (
        df["home_team_api"].str.contains(search, case=False, na=False)
        | df["away_team_api"].str.contains(search, case=False, na=False)
    )
    df = df[mask]
    if df.empty:
        st.info(f"Aucun match trouvé pour « {search} » sur cette sélection.")
        st.stop()

max_proba = df[["proba_H", "proba_D", "proba_A"]].max(axis=1)
nb_favoris = int((max_proba >= 0.55).sum())
st.markdown(
    f"""
    <div class="fp-kpis">
      <div class="fp-kpi"><div class="v">{len(df)}</div><div class="l">Matchs à venir</div></div>
      <div class="fp-kpi"><div class="v">{df['league'].nunique()}</div><div class="l">Compétitions</div></div>
      <div class="fp-kpi"><div class="v">{nb_favoris}</div><div class="l">Favoris nets (≥ 55%)</div></div>
      <div class="fp-kpi"><div class="v">{max_proba.mean() * 100:.0f}%</div><div class="l">Confiance moyenne</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="fp-legend">'
    '<span><span class="dot" style="background:var(--home)"></span>Victoire domicile</span>'
    '<span><span class="dot" style="background:var(--draw)"></span>Match nul</span>'
    '<span><span class="dot" style="background:var(--away)"></span>Victoire extérieure</span>'
    '</div>',
    unsafe_allow_html=True,
)

for match_date in sorted(df["date"].unique()):
    df_day = df[df["date"] == match_date]
    st.markdown(
        f'<div class="fp-date-header">{pd.Timestamp(match_date).strftime("%A %d %B %Y").capitalize()}</div>',
        unsafe_allow_html=True,
    )

    for league in df_day["league"].unique():
        df_league = df_day[df_day["league"] == league]
        st.markdown(f'<div class="fp-league-header">{LEAGUE_LABELS.get(league, league)}</div>', unsafe_allow_html=True)

        for _, row in df_league.iterrows():
            home_new = "" if row["home_team_known"] else '<span class="fp-new">nouveau</span>'
            away_new = "" if row["away_team_known"] else '<span class="fp-new">nouveau</span>'
            home_crest = row.get("home_crest", "")
            away_crest = row.get("away_crest", "")

            pct_h, pct_d, pct_a = row["proba_H"] * 100, row["proba_D"] * 100, row["proba_A"] * 100

            # Issue la plus probable -> pastille de confiance + équipe favorite mise en avant
            outcomes = {"H": pct_h, "D": pct_d, "A": pct_a}
            top_outcome = max(outcomes, key=outcomes.get)
            top_pct = outcomes[top_outcome]
            if top_pct >= 55:
                pill_class, pill_label = "strong", "Favori net"
            elif top_pct >= 40:
                pill_class, pill_label = "mid", "Tendance"
            else:
                pill_class, pill_label = "open", "Match ouvert"

            home_style = ' style="--favored-color:var(--home)"' if top_outcome == "H" else ""
            away_style = ' style="--favored-color:var(--away)"' if top_outcome == "A" else ""
            home_cls = "fp-team favored" if top_outcome == "H" else "fp-team"
            away_cls = "fp-team favored" if top_outcome == "A" else "fp-team"

            card = f"""
            <div class="fp-card">
              <div class="fp-teams-row">
                <span class="fp-pill {pill_class}">{pill_label} · {top_pct:.0f}%</span>
              </div>
              <div class="fp-teams">
                <div class="{home_cls}"{home_style}>
                  {f'<img src="{home_crest}">' if home_crest else ""}
                  {row['home_team_api']}{home_new}
                </div>
                <div class="fp-vs">vs</div>
                <div class="{away_cls}"{away_style}>
                  {row['away_team_api']}{away_new}
                  {f'<img src="{away_crest}">' if away_crest else ""}
                </div>
              </div>
              <div class="fp-bar">
                <div class="seg-h" style="width:{pct_h}%"></div>
                <div class="seg-d" style="width:{pct_d}%"></div>
                <div class="seg-a" style="width:{pct_a}%"></div>
              </div>
              <div class="fp-values">
                <span><span class="dot" style="background:var(--home)"></span>{pct_h:.0f}%</span>
                <span><span class="dot" style="background:var(--draw)"></span>{pct_d:.0f}%</span>
                <span><span class="dot" style="background:var(--away)"></span>{pct_a:.0f}%</span>
              </div>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)

if (~df["home_team_known"] | ~df["away_team_known"]).any():
    st.caption("« nouveau » = équipe sans historique dans nos données (promotion récente) — "
               "prédiction basée sur la moyenne de sa ligue, moins fiable que pour les autres équipes.")
