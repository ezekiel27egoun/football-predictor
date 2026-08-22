"""
Page d'administration — protégée par ADMIN_PASSWORD (.env / secrets).
Permet d'ajouter/renouveler un abonné : saisir son nom + numéro + les deux
dates (début/fin) achetées, le PIN est généré automatiquement -> à
copier-coller dans WhatsApp.

Accessible via le menu latéral de l'app ("Admin"), ou directement à
l'URL .../Admin.
"""
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from subscribers import add_or_renew_subscriber, extend_subscription, load_subscribers  # noqa: E402

load_dotenv()

st.set_page_config(page_title="Admin — Football Predictor", page_icon="🔐")
st.title("🔐 Administration des abonnés")

if "admin_ok" not in st.session_state:
    st.session_state.admin_ok = False

if not st.session_state.admin_ok:
    password = st.text_input("Mot de passe admin", type="password")
    if st.button("Se connecter"):
        if password == os.environ.get("ADMIN_PASSWORD"):
            st.session_state.admin_ok = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()

st.success("Connecté.")

st.subheader("Ajouter / renouveler un abonné")
with st.form("add_subscriber"):
    name = st.text_input("Nom de l'abonné")
    phone = st.text_input("Numéro de téléphone (avec indicatif, ex: 22990000000)")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Date de début", value=date.today())
    with col2:
        end_date = st.date_input("Date de fin", value=date.today() + timedelta(days=1))
    submitted = st.form_submit_button("Générer le PIN")

if submitted:
    if not phone or not name:
        st.error("Nom et numéro obligatoires.")
    elif end_date < start_date:
        st.error("La date de fin doit être après la date de début.")
    else:
        pin = add_or_renew_subscriber(phone.strip(), name.strip(), start_date, end_date)
        n_days = (end_date - start_date).days + 1
        st.success(
            f"PIN généré pour **{name}** ({phone}) : **{pin}** "
            f"— du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')} ({n_days} jour(s))"
        )
        st.info("Copie ce PIN et envoie-le à l'abonné par WhatsApp — c'est le seul moment où il est affiché.")

st.divider()
st.subheader("Prolonger un abonnement existant (même code)")
st.caption(
    "Pour un abonné qui paie AVANT l'expiration de sa période actuelle : "
    "repousse juste la date de fin, sans générer un nouveau PIN ni le "
    "déconnecter de ses appareils. À utiliser uniquement si l'abonnement "
    "n'a pas encore expiré — une fois expiré, utilisez plutôt la section "
    "ci-dessus (« Générer le PIN »), qui envoie automatiquement un nouveau "
    "code."
)
df_current = load_subscribers()
if df_current.empty:
    st.caption("Aucun abonné à prolonger pour l'instant.")
else:
    with st.form("extend_subscriber"):
        phone_to_extend = st.selectbox(
            "Abonné",
            options=df_current["phone"],
            format_func=lambda p: (
                f"{df_current.loc[df_current['phone'] == p, 'name'].values[0]} ({p}) — "
                f"expire le {df_current.loc[df_current['phone'] == p, 'expiry_date'].values[0]}"
            ),
        )
        new_end_date = st.date_input("Nouvelle date de fin", value=date.today() + timedelta(days=7))
        extend_submitted = st.form_submit_button("Prolonger (garder le même code)")

    if extend_submitted:
        ok = extend_subscription(phone_to_extend.strip(), new_end_date)
        if ok:
            st.success(
                f"Abonnement prolongé jusqu'au {new_end_date.strftime('%d/%m/%Y')} — "
                "le PIN de l'abonné n'a pas changé, rien à lui renvoyer."
            )
        else:
            st.error("Cet abonné n'a pas été trouvé.")

st.divider()
st.subheader("Abonnés actuels")
df = load_subscribers()
if df.empty:
    st.caption("Aucun abonné pour l'instant.")
else:
    st.dataframe(df[["name", "phone", "start_date", "expiry_date"]], use_container_width=True, hide_index=True)
