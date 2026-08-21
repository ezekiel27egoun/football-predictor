"""
App Streamlit : probabilités H/D/A pour les matchs à venir des 5 grands
championnats européens + Champions League.
"""
import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
  /* La couleur suit désormais la PROBABILITÉ, pas l'équipe (domicile/nul/
     extérieur) : vert = issue la plus probable, peu importe laquelle ;
     gris = les deux autres. L'identité (qui est qui) passe par le texte
     ("Dom./Nul/Ext.") et la position, jamais par la couleur seule. */
  --likely:          #1baf7a;
  --unlikely-fill:   #ded9d0;
  /* Purement décoratif (titre, liseré de l'avertissement) -> aucun lien
     avec la lecture des probabilités, jamais utilisé sur une barre/valeur. */
  --brand-accent:    #2a78d6;
  --warn-accent:     #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-fp-theme="light"]) {
    --surface-1:      #1a1a19;
    --page-plane:      #0d0d0d;
    --text-primary:    #ffffff;
    --text-secondary:  #c3c2b7;
    --border:          rgba(255,255,255,0.10);
    --likely:          #22c589;
    --unlikely-fill:   #3a3934;
    --brand-accent:    #3987e5;
    --warn-accent:     #d95926;
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

.fp-teams { display:flex; align-items:center; justify-content:space-between; margin-bottom: 10px; gap: 8px; }
.fp-team { display:flex; align-items:center; gap:8px; font-size: 15px; font-weight: 600; color: var(--text-primary);
  min-width: 0; flex: 1 1 auto; }
.fp-team img { width: 24px; height: 24px; object-fit: contain; flex-shrink: 0; }
.fp-team span.name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fp-vs { color: var(--text-muted); font-size: 12px; padding: 0 10px; flex-shrink: 0; }

/* Écran étroit (mobile) : les noms longs poussaient le "vs" et se
   mélangeaient visuellement avec les probabilités -> équipes empilées
   verticalement à la place, chacune sur toute la largeur disponible. */
@media (max-width: 480px) {
  .fp-teams { flex-direction: column; align-items: stretch; }
  .fp-team { justify-content: flex-start; }
  .fp-team.away { flex-direction: row-reverse; justify-content: flex-end; }
  .fp-vs { align-self: center; padding: 2px 0; }
}
.fp-new { font-size: 11px; color: var(--text-muted); font-weight: 400; margin-left: 4px; }

.fp-bar { display:flex; height: 16px; border-radius: 4px; overflow: hidden; background: var(--page-plane); }
.fp-bar div { height: 100%; }
.fp-bar .seg-top { background: var(--likely); }
.fp-bar .seg-rest { background: var(--unlikely-fill); }
.fp-bar div + div { margin-left: 2px; }

.fp-values { display:flex; justify-content:space-between; margin-top: 6px;
  font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-secondary);
  font-variant-numeric: tabular-nums; }
.fp-values .label { color: var(--text-muted); margin-right: 4px; }
.fp-values .top .label, .fp-values .top .pct { color: var(--text-primary); font-weight: 600; }

.fp-date-header { font: 700 20px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary); margin: 24px 0 4px 0; }

.fp-hero { margin: 4px 0 6px 0; }
.fp-hero h1 { font: 800 40px/1.15 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; letter-spacing: -0.01em;
  background: linear-gradient(90deg, var(--brand-accent) 0%, var(--likely) 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.fp-byline { font: 500 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-secondary); margin: 2px 0 18px 0; }
.fp-byline b { color: var(--text-primary); }

.fp-disclaimer { display:flex; gap:10px; align-items:flex-start; background: var(--surface-1);
  border: 1px solid var(--border); border-left: 3px solid var(--warn-accent); border-radius: 8px;
  padding: 10px 14px; margin: 0 0 22px 0;
  font: 13px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-secondary); }
.fp-disclaimer .icon { font-size: 16px; line-height: 1.5; }

.fp-kpis { display:flex; gap:12px; margin: 0 0 20px 0; flex-wrap: wrap; }
.fp-kpi { flex:1; min-width: 140px; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.fp-kpi .v { font: 700 24px/1.2 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-primary); }
.fp-kpi .l { font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-muted); }

.fp-pill { font: 600 11px/1; padding: 4px 8px; border-radius: 20px; white-space: nowrap; }
.fp-pill.strong { background: color-mix(in srgb, var(--likely) 18%, transparent); color: var(--likely); }
.fp-pill.mid { background: color-mix(in srgb, var(--likely) 8%, transparent); color: var(--likely); }
.fp-pill.open { background: var(--page-plane); color: var(--text-muted); border: 1px solid var(--border); }
.fp-pill.played { background: var(--page-plane); color: var(--text-secondary); border: 1px solid var(--border); }
.fp-teams-row { display:flex; align-items:center; justify-content:space-between; margin-bottom: 6px; }
.fp-kickoff { font: 600 12px/1; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.fp-team.favored { color: var(--likely); }

.fp-score { font: 700 20px/1; color: var(--text-primary); padding: 0 10px;
  font-variant-numeric: tabular-nums; }

.fp-week-nav { display:flex; align-items:center; gap:14px; margin: 0 0 18px 0; }
.fp-week-label { font: 600 15px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary); min-width: 220px; text-align:center; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# Fenêtre large fetchée UNE SEULE FOIS (mise en cache 1h) : navigation ensuite
# instantanée d'une semaine à l'autre, sans re-solliciter l'API à chaque clic.
WINDOW_PAST_DAYS = 200   # ~toute la saison en cours en arrière
WINDOW_FUTURE_DAYS = 90


@st.cache_data(ttl=3 * 3600)  # 3h de cache -> évite de re-solliciter l'API à chaque interaction
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

# --- Navigation par semaine (flèches) ---
# anchor_monday = lundi de la semaine affichée, gardé en session pour ne pas
# re-fetcher l'API à chaque clic (le fetch large est mis en cache 1h).
today = date.today()
if "anchor_monday" not in st.session_state:
    st.session_state.anchor_monday = today - timedelta(days=today.weekday())

nav_prev, nav_label, nav_next, nav_today = st.columns([1, 4, 1, 1.4])
with nav_prev:
    if st.button("◀ Semaine précédente", use_container_width=True):
        st.session_state.anchor_monday -= timedelta(days=7)
with nav_next:
    if st.button("Semaine suivante ▶", use_container_width=True):
        st.session_state.anchor_monday += timedelta(days=7)
with nav_today:
    if st.button("Aujourd'hui", use_container_width=True):
        st.session_state.anchor_monday = today - timedelta(days=today.weekday())

week_start = st.session_state.anchor_monday
week_end = week_start + timedelta(days=6)
with nav_label:
    st.markdown(
        f'<div class="fp-week-label">{week_start.strftime("%d %b").capitalize()} '
        f'→ {week_end.strftime("%d %b %Y").capitalize()}</div>',
        unsafe_allow_html=True,
    )

with st.spinner("Récupération des matchs (première fois seulement, ~1min)…"):
    df_wide = get_predictions(today - timedelta(days=WINDOW_PAST_DAYS), today + timedelta(days=WINDOW_FUTURE_DAYS))

if df_wide.empty:
    st.info("Aucun match trouvé pour les compétitions suivies.")
    st.stop()

# Heure de coup d'envoi en heure française (l'API renvoie de l'UTC) -> gère
# automatiquement CET/CEST, pas besoin de gérer le changement d'heure à la main.
df_wide["kickoff_paris"] = df_wide["kickoff_utc"].dt.tz_convert(ZoneInfo("Europe/Paris"))

df = df_wide[(df_wide["date"] >= pd.Timestamp(week_start)) & (df_wide["date"] <= pd.Timestamp(week_end))]

col3, = st.columns([1])
with col3:
    available_leagues = [lg for lg in LEAGUE_LABELS if lg in df_wide["league"].unique()]
    selected_leagues = st.multiselect(
        "Compétitions",
        options=available_leagues,
        default=available_leagues,
        format_func=lambda lg: LEAGUE_LABELS.get(lg, lg),
    )

df = df[df["league"].isin(selected_leagues)]
if df.empty:
    st.info("Aucun match cette semaine pour les compétitions sélectionnées.")
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

df_upcoming = df[df["status"] != "FINISHED"]
df_played = df[df["status"] == "FINISHED"]
max_proba = df_upcoming[["proba_H", "proba_D", "proba_A"]].max(axis=1)
nb_favoris = int((max_proba >= 0.55).sum())
confiance_moyenne = f"{max_proba.mean() * 100:.0f}%" if not df_upcoming.empty else "—"
st.markdown(
    f"""
    <div class="fp-kpis">
      <div class="fp-kpi"><div class="v">{len(df_upcoming)}</div><div class="l">Matchs à venir</div></div>
      <div class="fp-kpi"><div class="v">{len(df_played)}</div><div class="l">Matchs joués</div></div>
      <div class="fp-kpi"><div class="v">{df['league'].nunique()}</div><div class="l">Compétitions</div></div>
      <div class="fp-kpi"><div class="v">{nb_favoris}</div><div class="l">Favoris nets (≥ 55%)</div></div>
      <div class="fp-kpi"><div class="v">{confiance_moyenne}</div><div class="l">Confiance moyenne</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="fp-legend">'
    '<span><span class="dot" style="background:var(--likely)"></span>Issue la plus probable</span>'
    '<span><span class="dot" style="background:var(--unlikely-fill)"></span>Moins probable</span>'
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
            matchday_label = f" · J{int(row['matchday'])}" if pd.notna(row.get("matchday")) else ""
            kickoff_label = f"{row['kickoff_paris'].strftime('%H:%M')}{matchday_label}"

            if row["status"] == "FINISHED":
                # Match déjà joué -> score réel, pas de prédiction (inutile)
                h_score, a_score = int(row["home_score"]), int(row["away_score"])
                if h_score > a_score:
                    home_cls, away_cls = "fp-team favored", "fp-team away"
                elif a_score > h_score:
                    home_cls, away_cls = "fp-team", "fp-team away favored"
                else:
                    home_cls, away_cls = "fp-team", "fp-team away"

                card = f"""
                <div class="fp-card">
                  <div class="fp-teams-row">
                    <span class="fp-kickoff">{kickoff_label}</span>
                    <span class="fp-pill played">Terminé</span>
                  </div>
                  <div class="fp-teams">
                    <div class="{home_cls}">
                      {f'<img src="{home_crest}">' if home_crest else ""}
                      <span class="name">{row['home_team_api']}</span>{home_new}
                    </div>
                    <div class="fp-score">{h_score} - {a_score}</div>
                    <div class="{away_cls}">
                      <span class="name">{row['away_team_api']}</span>{away_new}
                      {f'<img src="{away_crest}">' if away_crest else ""}
                    </div>
                  </div>
                </div>
                """
                st.markdown(card, unsafe_allow_html=True)
                continue

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

            home_cls = "fp-team favored" if top_outcome == "H" else "fp-team"
            away_cls = "fp-team away favored" if top_outcome == "A" else "fp-team away"

            # Couleur = probabilité (vert = plus probable, gris = le reste),
            # jamais l'équipe -> identité portée par le texte (Dom./Nul/Ext.)
            # et la position, pas par la couleur (cf. décision produit).
            seg_cls = {k: ("seg-top" if k == top_outcome else "seg-rest") for k in ("H", "D", "A")}
            val_cls = {k: ("top" if k == top_outcome else "") for k in ("H", "D", "A")}

            card = f"""
            <div class="fp-card">
              <div class="fp-teams-row">
                <span class="fp-kickoff">{kickoff_label}</span>
                <span class="fp-pill {pill_class}">{pill_label} · {top_pct:.0f}%</span>
              </div>
              <div class="fp-teams">
                <div class="{home_cls}">
                  {f'<img src="{home_crest}">' if home_crest else ""}
                  <span class="name">{row['home_team_api']}</span>{home_new}
                </div>
                <div class="fp-vs">vs</div>
                <div class="{away_cls}">
                  <span class="name">{row['away_team_api']}</span>{away_new}
                  {f'<img src="{away_crest}">' if away_crest else ""}
                </div>
              </div>
              <div class="fp-bar">
                <div class="{seg_cls['H']}" style="width:{pct_h}%"></div>
                <div class="{seg_cls['D']}" style="width:{pct_d}%"></div>
                <div class="{seg_cls['A']}" style="width:{pct_a}%"></div>
              </div>
              <div class="fp-values">
                <span class="{val_cls['H']}"><span class="label">Dom.</span><span class="pct">{pct_h:.0f}%</span></span>
                <span class="{val_cls['D']}"><span class="label">Nul</span><span class="pct">{pct_d:.0f}%</span></span>
                <span class="{val_cls['A']}"><span class="label">Ext.</span><span class="pct">{pct_a:.0f}%</span></span>
              </div>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)

if (~df["home_team_known"] | ~df["away_team_known"]).any():
    st.caption("« nouveau » = équipe sans historique dans nos données (promotion récente) — "
               "prédiction basée sur la moyenne de sa ligue, moins fiable que pour les autres équipes.")
