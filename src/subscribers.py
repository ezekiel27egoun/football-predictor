"""
Gestion des abonnés payants (paywall) : liste stockée dans un Google Sheet
(éditable facilement depuis un téléphone, pas besoin d'une vraie base de
données à ce stade). Colonnes attendues dans la feuille : phone, pin,
expiry_date, device_tokens.

Variables d'environnement nécessaires (.env, jamais commitées) :
    GOOGLE_SERVICE_ACCOUNT_FILE : chemin vers le fichier JSON de la clé de
        service Google (compte de service avec accès à la feuille)
    SUBSCRIBERS_SHEET_ID : identifiant du Google Sheet (dans son URL)
    ADMIN_PASSWORD : mot de passe pour accéder à la page d'administration
"""
import os
import random
import string
from datetime import date, timedelta

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

MAX_DEVICES = 2  # nombre d'appareils actifs autorisés simultanément par abonné


def _get_worksheet():
    creds = Credentials.from_service_account_file(
        os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(os.environ["SUBSCRIBERS_SHEET_ID"])
    return sheet.sheet1


def load_subscribers():
    """Retourne la liste des abonnés sous forme de DataFrame (colonnes en texte)."""
    ws = _get_worksheet()
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["phone", "pin", "expiry_date", "device_tokens"])
    df["phone"] = df["phone"].astype(str)
    df["pin"] = df["pin"].astype(str)
    return df


def generate_pin():
    return "".join(random.choices(string.digits, k=4))


def add_or_renew_subscriber(phone, duration_days):
    """
    Ajoute un nouvel abonné, ou renouvelle un existant (nouveau PIN, nouvelle
    date d'expiration -> l'ancien PIN cesse de fonctionner). Retourne le PIN
    généré, à transmettre manuellement (WhatsApp) à l'abonné.
    """
    ws = _get_worksheet()
    df = load_subscribers()
    pin = generate_pin()
    expiry = (date.today() + timedelta(days=duration_days)).isoformat()

    existing = df.index[df["phone"] == str(phone)]
    if len(existing):
        row_number = existing[0] + 2  # +1 en-tête, +1 index 0-based -> 1-based
        ws.update(f"B{row_number}:D{row_number}", [[pin, expiry, ""]])  # réinitialise aussi les appareils
    else:
        ws.append_row([str(phone), pin, expiry, ""])

    return pin


def check_login(phone, pin):
    """Retourne (ok: bool, message_erreur: str|None)."""
    df = load_subscribers()
    match = df[(df["phone"] == str(phone)) & (df["pin"] == str(pin))]
    if match.empty:
        return False, "Numéro ou code incorrect."

    expiry = pd.to_datetime(match.iloc[0]["expiry_date"]).date()
    if date.today() > expiry:
        return False, "Abonnement expiré — contacte-nous pour le renouveler."
    return True, None


def is_subscription_active(phone):
    """
    Vérifie uniquement la date d'expiration (pas le PIN) -> utilisé pour un
    appareil déjà reconnu, pour re-vérifier à chaque visite que l'abonnement
    n'a pas expiré depuis (sinon un appareil resterait "reconnu" à vie).
    """
    df = load_subscribers()
    rows = df[df["phone"] == str(phone)]
    if rows.empty:
        return False
    expiry = pd.to_datetime(rows.iloc[0]["expiry_date"]).date()
    return date.today() <= expiry


def register_device(phone, device_id):
    """
    Enregistre cet appareil comme actif pour cet abonné. Au-delà de
    MAX_DEVICES, le plus ancien est retiré (déconnecté).
    """
    ws = _get_worksheet()
    df = load_subscribers()
    rows = df.index[df["phone"] == str(phone)]
    if not len(rows):
        return
    row_idx = rows[0]
    tokens = [t for t in str(df.iloc[row_idx].get("device_tokens", "") or "").split(",") if t]

    if device_id in tokens:
        return  # déjà enregistré, rien à faire

    tokens.append(device_id)
    tokens = tokens[-MAX_DEVICES:]  # ne garde que les plus récents
    ws.update_cell(row_idx + 2, 4, ",".join(tokens))


def is_device_active(phone, device_id):
    df = load_subscribers()
    rows = df[df["phone"] == str(phone)]
    if rows.empty:
        return False
    tokens = str(rows.iloc[0].get("device_tokens", "") or "").split(",")
    return device_id in tokens
