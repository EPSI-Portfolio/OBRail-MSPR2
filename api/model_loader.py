"""
model_loader.py
Rôle : Charger le modèle ML une seule fois au démarrage de l'API.
"""

import joblib
import os

# Chemin vers le modèle sauvegardé
MODEL_PATH = "models/model_underserved.joblib"


def load_model():
    """
    Charge le modèle ML depuis le fichier .joblib

    Returns:
        modèle entraîné (RandomForest ou XGBoost)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès")
    return model


# Chargement du modèle une seule fois (optimisation)
model = load_model()