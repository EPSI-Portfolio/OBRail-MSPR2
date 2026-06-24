# ObRail — Détection des sous-dessertes ferroviaires

Projet MSPR 2025-2026 (EPSI — Développeur en IA et Data Science, blocs E6.2 / E6.4).

ObRail Europe est un observatoire de la mobilité ferroviaire durable. Ce projet développe un modèle de **classification binaire** qui identifie les relations ferroviaires intercités européennes **sous-desservies** (`is_underserved`), afin d'aider les décideurs et opérateurs à repérer les zones fragiles du réseau.

La cible est définie par règle métier : une route est considérée sous-desservie si elle **circule 3 jours par semaine ou moins** **et** couvre une **distance supérieure à 100 km**.

---

## Structure du dépôt

```
data/
  raw/                 données sources (GTFS, Eurostat)
  processed/           données nettoyées + splits (X_train/X_test, y_train/y_test, scaler.joblib…)
src/                   pipeline Python (data_prep, features, train, optimize, evaluate, predict)
notebooks/             notebooks Jupyter (EDA → évaluation + analyses complémentaires)
models/                modèles sauvegardés (best_model_optimized.joblib = modèle retenu) + métadonnées
evaluation/
  plots/               figures produites par les notebooks
api/                   service FastAPI d'exposition du modèle
ethics/                rgpd_compliance.md, bias_analysis.md, limitations.md
docs/                  documentation technique, benchmark, veille
requirements.txt
```

---

## Pipeline des notebooks

Les notebooks s'exécutent dans l'ordre, chacun lisant les sorties du précédent. **Un notebook = un·e propriétaire** pour limiter les conflits de fusion.

### Pipeline principal

| Notebook | Contenu |
|---|---|
| `01_eda.ipynb` | Analyse exploratoire : qualité, distribution de la cible, relations features / cible |
| `02_feature_engineering.ipynb` | Construction et sélection des features, encodage, standardisation |
| `03_models.ipynb` | Modèles candidats : régression logistique, forêt aléatoire, LightGBM, MLP |
| `04_optimization.ipynb` | RandomizedSearchCV puis GridSearchCV — **LightGBM retenu** comme modèle de production |
| `05_explainability.ipynb` | Explicabilité du modèle retenu : SHAP (global) et LIME (prédictions individuelles) |
| `06_evaluation.ipynb` | Évaluation finale : métriques, matrice de confusion, courbes ROC |

### Notebooks complémentaires

| Notebook | Contenu |
|---|---|
| `07_cartographie.ipynb` | Carte choroplèthe du taux de sous-desserte par pays |
| `08_voting.ipynb` | VotingClassifier — assemblage des modèles (exploratoire) |
| `09_coherence_cible.ipynb` | Tests de cohérence statistique de la variable cible `is_underserved` |

---

## Modèles et résultats

Quatre modèles ont été testés : régression logistique, forêt aléatoire, LightGBM et MLP. **LightGBM** obtient le meilleur F1 et la meilleure stabilité en validation croisée ; c'est le modèle retenu (`models/best_model_optimized.joblib`).

| Métrique (jeu de test) | Valeur |
|---|---|
| Accuracy | 0,75 |
| F1 (classe sous-desservie) | 0,60 |
| ROC-AUC | 0,88 |
| Recall (sous-desservie) | 0,97 |
| Precision (sous-desservie) | 0,44 |

Le déséquilibre des classes (19,4 % de cas positifs) est traité via `scale_pos_weight` (≈ 4,16). Le recall élevé est privilégié : dans une logique d'aide à la décision, mieux vaut signaler une ligne à tort que d'en manquer une réellement sous-desservie.

---

## Installation

```bash
git clone <repo>
cd OBRail-MSPR2
pip install -r requirements.txt
```

---

## Lancer les notebooks

```bash
jupyter lab        # ou : jupyter notebook
```

Ouvrir les notebooks dans l'ordre (`01` → `06`) et exécuter **Restart & Run All**. Les figures sont enregistrées dans `evaluation/plots/`.

---

## API de prédiction

L'API FastAPI expose le modèle retenu et prédit le statut de sous-desserte d'une liaison.

### Lancer le service

```bash
uvicorn api.main:app --reload
```

Documentation interactive (Swagger) : `http://127.0.0.1:8000/docs`

### Endpoints

- `POST /predict` — prédiction du statut de sous-desserte
- `GET /health` — disponibilité du service et bon chargement du modèle

### Exemple de requête

```json
{
  "departure_country": "FR",
  "service_type": "night",
  "distance_km": 450.0,
  "co2_per_pkm": 28.3,
  "arrival_country": "DE"
}
```

### Exemple de réponse

```json
{
  "is_underserved": 1,
  "probability": 0.8444,
  "label": "sous-desservie"
}
```

Le pipeline de prédiction (`src/predict.py`, fonction `predict_route()`) réalise la transformation des entrées (log-transformations, encodage one-hot du pays, calcul de `is_international`) avant la prédiction. C'est le point d'entrée unique utilisé par l'API.

---

## Reproductibilité

- `random_state = 42` partout.
- Chemins relatifs : lecture dans `data/...`, écriture des figures dans `evaluation/plots/`.
- Modèle sauvegardé : `models/best_model_optimized.joblib`, chargé par `05`, `06`, `src/evaluate.py` et `src/predict.py`.

---

## Limites

- Dépendance du modèle à `log_distance` : la cible étant définie en partie sur la distance, le modèle apprend surtout ce seuil (circularité partielle, documentée dans `ethics/limitations.md`).
- Taux de faux positifs élevé (précision 0,44), lié au déséquilibre des classes.
- Déséquilibre géographique : France, Espagne et Allemagne représentent une large part des données.
- Certaines variables (CO2) partiellement simulées et peu informatives.

---

## Perspectives

- Construire une cible fondée sur des données réelles de fréquence de desserte plutôt qu'une règle métier.
- Enrichir le modèle avec des variables temporelles et géographiques supplémentaires.
- Déploiement cloud (Azure Machine Learning identifié comme piste) avec réentraînement régulier.

---

## Équipe

Charlotte, Jeannette, Louis, Sammy.