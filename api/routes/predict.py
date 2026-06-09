"""
predict.py
Rôle : Endpoint de prédiction /predict
"""

from fastapi import APIRouter
import numpy as np

from api.schemas import PredictionInput, PredictionOutput
from api.model_loader import model

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

    # Conversion des données en tableau pour le modèle
    features = np.array([[
        data.distance_km,
        data.type_encoded,
        data.is_cross_border,
        data.passengers_estimated,
        data.capacity,
        data.load_factor
    ]])

    # Prédiction avec le modèle
    prediction = model.predict(features)[0]

    return {"is_underserved": int(prediction)}