"""
Porte d'accès abonné : numéro + PIN, avec appareil "mémorisé" via un témoin
(cookie) pour ne pas redemander le code à chaque visite -> seulement sur un
nouvel appareil, ou une fois l'abonnement expiré / renouvelé (nouveau PIN).
"""
import uuid

import streamlit as st
from extra_streamlit_components import CookieManager

from subscribers import check_login, is_device_active, is_subscription_active, register_device

COOKIE_DEVICE_KEY = "fp_device_id"
COOKIE_PHONE_KEY = "fp_phone"


def require_subscription():
    """
    Retourne True si l'utilisateur est un abonné actif (appareil déjà
    reconnu, ou vient de saisir un numéro + PIN valides) -> affiche le
    formulaire de connexion et retourne False sinon.
    """
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
        return True

    st.markdown("### 🔒 Accès abonné")
    st.caption("Entre le numéro et le code reçus par WhatsApp lors de ton abonnement.")
    with st.form("login_form"):
        phone_input = st.text_input("Numéro de téléphone")
        pin_input = st.text_input("Code PIN", max_chars=4)
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        ok, error = check_login(phone_input.strip(), pin_input.strip())
        if ok:
            register_device(phone_input.strip(), device_id)
            cookie_manager.set(COOKIE_PHONE_KEY, phone_input.strip())
            st.rerun()
        else:
            st.error(error)

    return False
