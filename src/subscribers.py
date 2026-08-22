"""
Gestion des abonnés payants (paywall) : liste stockée dans un Google Sheet
(éditable facilement depuis un téléphone, pas besoin d'une vraie base de
données à ce stade). Colonnes attendues dans la feuille (dans cet ordre) :
phone, name, pin, start_date, expiry_date, device_tokens.

Variables d'environnement nécessaires (.env en local, secrets Streamlit Cloud
en ligne -> jamais commitées) :
    GOOGLE_SERVICE_ACCOUNT_FILE : chemin vers le fichier JSON de la clé de
        service Google (usage local -> le fichier existe sur le disque)
    GOOGLE_SERVICE_ACCOUNT_JSON : contenu JSON de la clé, en clair (usage
        Streamlit Cloud -> pas de fichier possible, collé comme secret) ;
        prioritaire sur GOOGLE_SERVICE_ACCOUNT_FILE s'il est présent
    SUBSCRIBERS_SHEET_ID : identifiant du Google Sheet (dans son URL)
    ADMIN_PASSWORD : mot de passe pour accéder à la page d'administration
"""
import json
import os
import random
import string
from datetime import date
from pathlib import Path

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_DEVICES = 2  # nombre d'appareils actifs autorisés simultanément par abonné
COLUMNS = ["phone", "name", "pin", "start_date", "expiry_date", "device_tokens"]
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _service_account_path():
    """Chemin absolu vers le fichier de clé -> fonctionne peu importe le
    répertoire courant depuis lequel le script/l'app est lancé (cohérent
    avec le reste du projet, ex: predict_matches.py)."""
    raw = os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"]
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _get_credentials():
    """
    En local : fichier .json sur le disque (GOOGLE_SERVICE_ACCOUNT_FILE).
    Sur Streamlit Cloud : pas de fichier possible -> le contenu JSON collé
    tel quel comme secret (GOOGLE_SERVICE_ACCOUNT_JSON).
    """
    raw_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw_json:
        return Credentials.from_service_account_info(json.loads(raw_json), scopes=SCOPES)
    return Credentials.from_service_account_file(str(_service_account_path()), scopes=SCOPES)


def _get_worksheet():
    creds = _get_credentials()
    client = gspread.authorize(creds)
    sheet = client.open_by_key(os.environ["SUBSCRIBERS_SHEET_ID"])
    return sheet.sheet1


def load_subscribers():
    """Retourne la liste des abonnés sous forme de DataFrame (colonnes en texte)."""
    ws = _get_worksheet()
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["phone"] = df["phone"].astype(str)
    df["pin"] = df["pin"].astype(str)
    return df


def generate_pin():
    return "".join(random.choices(string.digits, k=4))


def add_or_renew_subscriber(phone, name, start_date, end_date):
    """
    Ajoute un nouvel abonné, ou renouvelle un existant (nouveau PIN, nouvelle
    période -> l'ancien PIN cesse de fonctionner). start_date/end_date :
    objets date. Retourne le PIN généré, à transmettre manuellement
    (WhatsApp) à l'abonné.
    """
    ws = _get_worksheet()
    df = load_subscribers()
    pin = generate_pin()
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    existing = df.index[df["phone"] == str(phone)]
    if len(existing):
        row_number = existing[0] + 2  # +1 en-tête, +1 index 0-based -> 1-based
        # B:F -> name, pin, start_date, expiry_date, device_tokens (réinitialisés)
        ws.update(f"B{row_number}:F{row_number}", [[name, pin, start_iso, end_iso, ""]])
    else:
        ws.append_row([str(phone), name, pin, start_iso, end_iso, ""])

    return pin


def extend_subscription(phone, new_end_date):
    """
    Prolonge un abonné déjà existant SANS régénérer son PIN ni réinitialiser
    ses appareils reconnus — contrairement à add_or_renew_subscriber(), qui
    est prévue pour le cas "expiré, nouveau code à renvoyer". Utile pour
    prolonger avant expiration, sans déconnecter l'abonné ni lui redemander
    un nouveau code. Ne touche que la colonne expiry_date (E). Retourne
    True si l'abonné existait et a été mis à jour, False sinon.
    """
    ws = _get_worksheet()
    df = load_subscribers()
    existing = df.index[df["phone"] == str(phone)]
    if not len(existing):
        return False
    row_number = existing[0] + 2  # +1 en-tête, +1 index 0-based -> 1-based
    ws.update(f"E{row_number}", [[new_end_date.isoformat()]])
    return True


def _subscription_window(row):
    start = pd.to_datetime(row["start_date"]).date()
    end = pd.to_datetime(row["expiry_date"]).date()
    return start, end


def check_login(phone, pin):
    """Retourne (ok: bool, message_erreur: str|None)."""
    df = load_subscribers()
    match = df[(df["phone"] == str(phone)) & (df["pin"] == str(pin))]
    if match.empty:
        return False, "Numéro ou code incorrect."

    start, end = _subscription_window(match.iloc[0])
    today = date.today()
    if today < start:
        return False, f"Abonnement pas encore actif (à partir du {start.strftime('%d/%m/%Y')})."
    if today > end:
        return False, "Abonnement expiré — contactez-nous pour le renouveler."
    return True, None


def is_subscription_active(phone):
    """
    Vérifie uniquement les dates (pas le PIN) -> utilisé pour un appareil
    déjà reconnu, pour re-vérifier à chaque visite que l'abonnement est
    toujours dans sa période valide (sinon un appareil resterait "reconnu"
    même après expiration).
    """
    df = load_subscribers()
    rows = df[df["phone"] == str(phone)]
    if rows.empty:
        return False
    start, end = _subscription_window(rows.iloc[0])
    return start <= date.today() <= end


def get_expiry_date(phone):
    """Date de fin d'abonnement de cet abonné, ou None si introuvable."""
    df = load_subscribers()
    rows = df[df["phone"] == str(phone)]
    if rows.empty:
        return None
    _, end = _subscription_window(rows.iloc[0])
    return end


def get_subscription_window(phone):
    """(date_debut, date_fin) de cet abonné, ou (None, None) si introuvable."""
    df = load_subscribers()
    rows = df[df["phone"] == str(phone)]
    if rows.empty:
        return None, None
    return _subscription_window(rows.iloc[0])


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
    ws.update_cell(row_idx + 2, 6, ",".join(tokens))


def is_device_active(phone, device_id):
    df = load_subscribers()
    rows = df[df["phone"] == str(phone)]
    if rows.empty:
        return False
    tokens = str(rows.iloc[0].get("device_tokens", "") or "").split(",")
    return device_id in tokens
