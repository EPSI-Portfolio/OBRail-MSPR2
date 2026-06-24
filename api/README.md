# Dossier `api/` — Service de prédiction ObRail

API REST développée avec **FastAPI**. Elle expose le modèle LightGBM retenu et prédit, pour une liaison ferroviaire donnée, si elle est **sous-desservie** (`is_underserved`).

L'API ne ré-implémente pas la logique de prédiction : elle importe la fonction `predict_route()` du pipeline `src/predict.py`, garantissant que l'API et les scripts utilisent exactement le même traitement (log-transformations, encodage one-hot du pays, calcul de `is_international`) et le même modèle.

---

## Prérequis

Le modèle entraîné et le scaler doivent être présents :

- `models/best_model_optimized.joblib`
- `data/processed/scaler.joblib`

S'ils n'existent pas, lancer d'abord le pipeline (voir `src/README.md`, étapes 1 à 4).

```bash
pip install -r requirements.txt
```

---

## Lancer le service

Depuis la **racine du projet** :

```bash
uvicorn api.main:app --reload
```

- API : `http://127.0.0.1:8000`
- Documentation interactive (Swagger) : `http://127.0.0.1:8000/docs`
- Schéma OpenAPI : `http://127.0.0.1:8000/openapi.json`

---

## Endpoints

### `GET /health`

Vérifie que le service répond et que le modèle est bien chargé en mémoire.

Réponse :

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /predict`

Prédit le statut de sous-desserte d'une liaison.

**Corps de la requête (PredictionInput)**

| Champ | Type | Obligatoire | Exemple |
|---|---|---|---|
| `departure_country` | string | Oui | `"FR"` |
| `service_type` | string | Oui | `"day"` ou `"night"` |
| `distance_km` | float | Oui | `450.0` |
| `co2_per_pkm` | float | Oui | `28.3` |
| `arrival_country` | string | Non | `"DE"` |

**Exemple de requête**

```json
{
  "departure_country": "FR",
  "service_type": "night",
  "distance_km": 450.0,
  "co2_per_pkm": 28.3,
  "arrival_country": "DE"
}
```

**Réponse (PredictionOutput)**

| Champ | Type | Description |
|---|---|---|
| `is_underserved` | int | `1` = sous-desservie, `0` = desservie |
| `probability` | float | Probabilité de la classe « sous-desservie » |
| `label` | string | Libellé lisible (`"sous-desservie"` / `"desservie"`) |
| `inputs` | object | Rappel des entrées reçues |

```json
{
  "is_underserved": 1,
  "probability": 0.8444,
  "label": "sous-desservie",
  "inputs": { "departure_country": "FR", "service_type": "night", "distance_km": 450.0, "co2_per_pkm": 28.3, "arrival_country": "DE" }
}
```

---

## Tester rapidement

Via `curl` :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"departure_country":"FR","service_type":"night","distance_km":450.0,"co2_per_pkm":28.3,"arrival_country":"DE"}'
```

Ou directement depuis l'interface Swagger (`/docs`), bouton **Try it out**.

---

## Organisation du dossier

```
api/
  main.py            point d'entrée FastAPI (création de l'app, montage des routes)
  routes/
    predict.py       endpoint POST /predict
  schemas.py         modèles Pydantic (PredictionInput, PredictionOutput)
  model_loader.py    chargement du modèle et du scaler au démarrage
```

> Adapter cette arborescence si la structure réelle du dossier diffère.

---

## Monitoring recommandé en production

- Taux de routes prédites comme sous-desservies (détection de dérive / data drift).
- Distribution des probabilités retournées (confiance moyenne du modèle).
- Latence de l'endpoint `/predict`.
- Nombre de requêtes contenant un pays inconnu du modèle (signal de réentraînement).