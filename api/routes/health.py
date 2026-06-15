"""
health.py
Rôle : Vérifier que l'API est en ligne
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    """
    Endpoint GET /health

    Permet de vérifier rapidement si l'API tourne
    """
    return {"status": "ok"}