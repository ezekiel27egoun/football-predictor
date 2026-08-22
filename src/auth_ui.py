"""
Porte d'accès abonné : numéro + PIN, avec appareil "mémorisé" via un témoin
(cookie) pour ne pas redemander le code à chaque visite -> seulement sur un
nouvel appareil, ou une fois l'abonnement expiré / renouvelé (nouveau PIN).
"""
import uuid

import streamlit as st
from extra_streamlit_components import CookieManager

from subscribers import (
    check_login,
    get_subscription_window,
    is_device_active,
    is_subscription_active,
    register_device,
)

COOKIE_DEVICE_KEY = "fp_device_id"
COOKIE_PHONE_KEY = "fp_phone"


def require_subscription():
    """
    Retourne True si l'utilisateur est un abonné actif (appareil déjà
    reconnu, ou vient de saisir un numéro + PIN valides) -> affiche le
    formulaire de connexion et retourne False sinon.
    """
    # Le cookie posé par CookieManager met un instant à se synchroniser côté
    # navigateur (aller-retour du composant) -> juste après une connexion
    # réussie, le relire immédiatement peut encore renvoyer l'ancienne
    # valeur (rien ne semble se passer). st.session_state, lui, est
    # disponible immédiatement -> fait foi pour la session EN COURS ; le
    # cookie ne sert qu'à reconnaître l'appareil lors d'une PROCHAINE visite.
    if st.session_state.get("fp_authenticated"):
        # Auto-réparation : si la session a été authentifiée par une version
        # antérieure du code (avant l'ajout des dates), fp_start_date/
        # fp_expiry_date peuvent manquer alors que fp_authenticated est resté
        # True -> aucune limite ne serait jamais appliquée. On les recalcule
        # ici si absentes, à partir du numéro déjà connu. Si même le numéro
        # est manquant, on force une reconnexion propre plutôt que de rester
        # authentifié sans savoir de qui il s'agit.
        if "fp_expiry_date" not in st.session_state:
            if st.session_state.get("fp_phone"):
                st.session_state.fp_start_date, st.session_state.fp_expiry_date = get_subscription_window(
                    st.session_state.fp_phone
                )
            else:
                st.session_state.fp_authenticated = False
                return False
        return True

    cookie_manager = CookieManager(key="fp_cookies")
    device_id = cookie_manager.get(COOKIE_DEVICE_KEY)
    if device_id is None:
        device_id = str(uuid.uuid4())
        cookie_manager.set(COOKIE_DEVICE_KEY, device_id)

    phone = cookie_manager.get(COOKIE_PHONE_KEY)

    # Appareil déjà reconnu pour ce numéro -> pas besoin de redemander le PIN,
    # tant que l'abonnement est toujours valide (vérifié à chaque visite,
    # pas juste au moment de la connexion).
    if phone and is_device_active(phone, device_id) and is_subscription_active(phone):
        st.session_state.fp_authenticated = True
        st.session_state.fp_phone = phone
        st.session_state.fp_start_date, st.session_state.fp_expiry_date = get_subscription_window(phone)
        return True

    st.markdown("### 🔒 Accès abonné")
    st.caption("Entrez le numéro et le code reçus par WhatsApp lors de votre abonnement.")
    with st.form("login_form"):
        phone_input = st.text_input("Numéro de téléphone")
        pin_input = st.text_input("Code PIN", max_chars=4)
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        ok, error = check_login(phone_input.strip(), pin_input.strip())
        if ok:
            register_device(phone_input.strip(), device_id)
            cookie_manager.set(COOKIE_PHONE_KEY, phone_input.strip())
            st.session_state.fp_authenticated = True
            st.session_state.fp_phone = phone_input.strip()
            st.session_state.fp_start_date, st.session_state.fp_expiry_date = get_subscription_window(
                phone_input.strip()
            )
            st.rerun()
        else:
            st.error(error)

    return False
