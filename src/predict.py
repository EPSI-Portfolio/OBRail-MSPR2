"""
predict.py — Pipeline de prédiction reproductible (ObRail MSPR2)

Charge le modèle LightGBM optimisé et le scaler, et expose predict_route()
C'est cette fonction que l'API REST (route /predict) doit appeler.

Modèle      : models/best_model_optimized.joblib
Scaler      : data/processed/scaler.joblib
Métadonnées : models/model_metadata.json

Features attendues (38 au total) :
    Numériques (normalisées) : rail_modal_share, log_distance, log_co2
    Binaires                 : type_encoded, is_international
    One-hot pays (×33)       : country_AT, country_BE, ... country_UA

Usage en ligne de commande :
    python src/predict.py
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ─── Chemins ────────────────────────────────────────────────────────────────

def find_project_root() -> Path:
    cwd = Path(__file__).resolve().parent
    for folder in [cwd] + list(cwd.parents):
        if (folder / 'data').exists() and (folder / 'models').exists():
            return folder
    raise FileNotFoundError('Racine du projet introuvable.')

ROOT        = find_project_root()
MODEL_PATH  = ROOT / 'models' / 'best_model_optimized.joblib'
SCALER_PATH = ROOT / 'data' / 'processed' / 'scaler.joblib'
META_PATH   = ROOT / 'models' / 'model_metadata.json'

# ─── Part modale ferroviaire par pays (source : Eurostat) ───────────────────

RAIL_MODAL = {
    "FR": 11.1, "DE": 9.9,  "DK": 7.9,  "GB": 9.2,  "IT": 6.8,  "ES": 6.5,
    "PL": 8.1,  "NL": 11.5, "BE": 8.3,  "AT": 12.1, "SE": 9.8,  "CH": 17.9,
    "CZ": 9.1,  "HU": 9.3,  "SK": 8.7,  "PT": 5.2,  "GR": 2.1,  "RO": 5.8,
    "HR": 4.3,  "SI": 4.8,  "FI": 5.9,  "NO": 6.2,  "IE": 3.8,  "LU": 7.1,
    "LT": 4.2,  "EE": 4.5,  "LV": 5.1,  "BG": 4.9,  "RS": 3.2,  "ME": 2.8,
    "MK": 2.1,  "AL": 1.9,  "UA": 6.8,  "MD": 4.1,  "TR": 5.0,
}

# Liste des colonnes one-hot pays dans l'ordre exact du training
# (drop_first=True sur sorted countries — AL est la référence supprimée)
COUNTRY_COLS = [
    'country_AT', 'country_BE', 'country_BG', 'country_CH', 'country_CZ',
    'country_DE', 'country_DK', 'country_EE', 'country_ES', 'country_FI',
    'country_FR', 'country_GB', 'country_GR', 'country_HR', 'country_HU',
    'country_IE', 'country_IT', 'country_LT', 'country_LU', 'country_MD',
    'country_ME', 'country_MK', 'country_NL', 'country_NO', 'country_PL',
    'country_PT', 'country_RO', 'country_RS', 'country_SE', 'country_SI',
    'country_SK', 'country_TR', 'country_UA',
]

COLS_TO_SCALE = ['rail_modal_share', 'log_distance', 'log_co2']

# ─── Chargement paresseux (une seule fois) ───────────────────────────────────

_model  = None
_scaler = None
_meta   = None


def _load():
    global _model, _scaler, _meta
    if _model is None:
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        with open(META_PATH, 'r') as f:
            _meta = json.load(f)
    return _model, _scaler, _meta


# ─── Fonction principale ─────────────────────────────────────────────────────

def predict_route(
    departure_country: str,
    service_type: str,
    distance_km: float,
    co2_per_pkm: float,
    arrival_country: str = None,
) -> dict:
    """
    Prédit si une liaison ferroviaire est sous-desservie.

    Args:
        departure_country : code pays ISO-2 ('FR', 'DE', 'ES'...)
        service_type      : 'day' ou 'night'
        distance_km       : distance de la liaison en kilomètres
        co2_per_pkm       : émissions CO2 par passager-km (en g)
        arrival_country   : code pays ISO-2 du pays d'arrivée (optionnel)
                            Si différent de departure_country → route internationale

    Returns:
        dict avec :
            is_underserved  : 0 ou 1
            probability     : probabilité d'être sous-desservie (0-1)
            inputs          : les valeurs utilisées pour la prédiction
    """
    model, scaler, meta = _load()

    # ── Validation des entrées ───────────────────────────────────────────────
    departure_country = departure_country.upper().strip()
    service_type      = service_type.lower().strip()

    if departure_country not in RAIL_MODAL:
        raise ValueError(
            f"Pays inconnu : '{departure_country}'. "
            f"Pays supportés : {sorted(RAIL_MODAL.keys())}"
        )
    if service_type not in ('day', 'night'):
        raise ValueError(
            f"service_type invalide : '{service_type}'. Attendu : 'day' ou 'night'"
        )
    if distance_km <= 0:
        raise ValueError("distance_km doit être positif")
    if co2_per_pkm <= 0:
        raise ValueError("co2_per_pkm doit être positif")

    # ── Construction des features ────────────────────────────────────────────
    is_international = 0
    if arrival_country is not None:
        is_international = int(
            arrival_country.upper().strip() != departure_country
        )

    rail_modal_share = RAIL_MODAL[departure_country]
    type_encoded     = 1 if service_type == 'night' else 0
    log_distance     = float(np.log1p(distance_km))
    log_co2          = float(np.log1p(co2_per_pkm))

    # One-hot pays
    country_values = {col: 0 for col in COUNTRY_COLS}
    col_name = f'country_{departure_country}'
    if col_name in country_values:
        country_values[col_name] = 1
    # Si le pays est AL (référence supprimée par drop_first), toutes les
    # colonnes one-hot restent à 0 — comportement correct

    # ── Assemblage du dataframe de features ─────────────────────────────────
    row = {
        'rail_modal_share': rail_modal_share,
        'type_encoded':     type_encoded,
        'is_international': is_international,
        'log_distance':     log_distance,
        'log_co2':          log_co2,
        **country_values,
    }

    X = pd.DataFrame([row])

    # ── Normalisation (uniquement les colonnes continues) ────────────────────
    X[COLS_TO_SCALE] = scaler.transform(X[COLS_TO_SCALE])

    # ── Prédiction ───────────────────────────────────────────────────────────
    proba         = float(model.predict_proba(X)[0, 1])
    is_underserved = int(proba >= 0.5)

    return {
        'is_underserved': is_underserved,
        'probability':    round(proba, 4),
        'label':          'sous-desservie' if is_underserved else 'desservie',
        'inputs': {
            'departure_country': departure_country,
            'arrival_country':   arrival_country,
            'service_type':      service_type,
            'distance_km':       distance_km,
            'co2_per_pkm':       co2_per_pkm,
            'is_international':  is_international,
        },
    }


# ─── Démonstration en ligne de commande ─────────────────────────────────────

if __name__ == '__main__':
    print('Démonstration du pipeline de prédiction ObRail')
    print('=' * 55)

    exemples = [
        {
            'departure_country': 'FR',
            'service_type':      'day',
            'distance_km':       450.0,
            'co2_per_pkm':       28.3,
            'arrival_country':   'FR',
            'description':       'Liaison de jour France — longue distance',
        },
        {
            'departure_country': 'DE',
            'service_type':      'night',
            'distance_km':       800.0,
            'co2_per_pkm':       29.1,
            'arrival_country':   'AT',
            'description':       'Train de nuit international DE→AT',
        },
        {
            'departure_country': 'ES',
            'service_type':      'day',
            'distance_km':       80.0,
            'co2_per_pkm':       27.5,
            'arrival_country':   'ES',
            'description':       'Liaison courte de jour Espagne',
        },
        {
            'departure_country': 'NL',
            'service_type':      'day',
            'distance_km':       200.0,
            'co2_per_pkm':       30.0,
            'arrival_country':   'BE',
            'description':       'Liaison internationale NL→BE',
        },
    ]

    for ex in exemples:
        desc = ex.pop('description')
        result = predict_route(**ex)
        statut = result['label'].upper()
        print(f"\n{desc}")
        print(f"  → {statut} (probabilité : {result['probability']})")
        print(f"     is_underserved = {result['is_underserved']}")