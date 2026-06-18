"""
predict.py
Rôle : Endpoint de prédiction /predict
Auteur : Jeannette
"""

from fastapi import APIRouter
from api.schemas import PredictionInput, PredictionOutput
from src.predict import predict_route

# ✅ CRÉATION DU ROUTER (OBLIGATOIRE)
router = APIRouter()


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    """
    Endpoint POST /predict

    Prend en entrée les caractéristiques d'une ligne ferroviaire
    et renvoie une prédiction :
    1 = sous-desservie
    0 = non sous-desservie
    """
    result = predict_route(
        departure_country=data.departure_country,
        service_type=data.service_type,
        distance_km=data.distance_km,
        co2_per_pkm=data.co2_per_pkm,
        arrival_country=data.arrival_country,
    )
    return result