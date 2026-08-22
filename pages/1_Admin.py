"""
Page d'administration — protégée par ADMIN_PASSWORD (.env / secrets).
Permet d'ajouter/renouveler un abonné : saisir son numéro + la durée
achetée, le PIN est généré automatiquement -> à copier-coller dans WhatsApp.

Accessible via le menu latéral de l'app ("Admin"), ou directement à
l'URL .../Admin.
"""
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from subscribers import add_or_renew_subscriber, load_subscribers  # noqa: E402

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
    phone = st.text_input("Numéro de téléphone (avec indicatif, ex: 22990000000)")
    duration = st.selectbox("Durée", options=[("Jour", 1), ("Semaine", 7), ("Mois", 30)],
                             format_func=lambda x: x[0])
    submitted = st.form_submit_button("Générer le PIN")

if submitted and phone:
    pin = add_or_renew_subscriber(phone.strip(), duration[1])
    st.success(f"PIN généré pour {phone} : **{pin}** (valable {duration[0].lower()})")
    st.info("Copie ce PIN et envoie-le à l'abonné par WhatsApp — c'est le seul moment où il est affiché.")

st.divider()
st.subheader("Abonnés actuels")
df = load_subscribers()
if df.empty:
    st.caption("Aucun abonné pour l'instant.")
else:
    st.dataframe(df[["phone", "expiry_date"]], use_container_width=True, hide_index=True)
