"""
schemas.py
Rôle : Définir les structures de données d'entrée et sortie de l'API.
"""

from pydantic import BaseModel


class PredictionInput(BaseModel):
    """
    Données attendues pour faire une prédiction
    """

    distance_km: float
    type_encoded: int
    is_cross_border: int
    passengers_estimated: int
    capacity: int
    load_factor: float


class PredictionOutput(BaseModel):
    """
    Réponse renvoyée par l'API
    """

    is_underserved: int