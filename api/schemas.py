"""
schemas.py
Rôle : Définir les structures de données d'entrée et sortie de l'API.
"""

from pydantic import BaseModel
from typing import Optional


class PredictionInput(BaseModel):
    """
    Données attendues pour faire une prédiction
    """
    departure_country: str
    service_type: str
    distance_km: float
    co2_per_pkm: float
    arrival_country: Optional[str] = None


class PredictionOutput(BaseModel):
    """
    Réponse renvoyée par l'API
    """
    is_underserved: int
    probability: float
    label: str
    inputs: dict