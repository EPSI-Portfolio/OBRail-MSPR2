# Rapport final — ObRail

## 1. Contexte
ObRail est un projet de Data Science visant à détecter les lignes ferroviaires sous-desservies, c’est-à-dire les liaisons où la demande de transport dépasse l’offre disponible. L’objectif est de fournir un outil opérationnel permettant d’identifier ces lignes à partir de données de transport, d’émissions et de caractéristiques de service.

## 2. Objectif du projet
- Développer une chaîne de traitement complète pour préparer les données ferroviaires.
- Construire des modèles de machine learning capables de classer une liaison comme sous-desservie ou non.
- Exposer une API REST simple pour permettre des prédictions en temps réel.
- Produire une documentation métier et technique claire pour assurer la reproductibilité.

## 3. Données utilisées
Le projet s’appuie sur un jeu de données ferroviaires enrichi avec :
- des informations de distance,
- des caractéristiques de service (jour/nuit),
- des émissions de CO2 par passager-kilomètre,
- des variables de fréquence et de charge.

Le dataset final a été nettoyé et transformé dans `data/processed/`, avec notamment :
- `ml_dataset.csv`,
- `X_train.csv` / `X_test.csv`,
- `y_train.csv` / `y_test.csv`,
- `scaler.joblib`.

## 4. Méthodologie
### 4.1 Prétraitement
- Nettoyage des données brutes.
- Sélection des variables pertinentes.
- Calcul de variables dérivées : `log_distance`, `log_co2`, `rail_modal_share`, `is_international`.
- Encodage catégoriel des pays et du type de service.
- Normalisation des variables continues avec un `StandardScaler`.

### 4.2 Feature engineering
Les features retenues comprennent :
- la part modale ferroviaire du pays de départ,
- la distance de la liaison,
- les émissions CO2 par passager-km,
- le type de service (`day` ou `night`),
- la nature transfrontalière ou nationale de la liaison,
- l’encodage one-hot du pays de départ.

### 4.3 Modèles testés
- Random Forest,
- XGBoost,
- LightGBM (utilisé dans le pipeline de production),
- Comparaison d’hyperparamètres via recherche d’optimisation.

## 5. Solution retenue
Le modèle final est un modèle optimisé entraîné localement et stocké dans `models/best_model_optimized.joblib`. Le pipeline de prédiction est implémenté dans `src/predict.py`.

Cette solution interne a été retenue pour :
- la maîtrise totale du pipeline,
- la transparence des choix techniques,
- la reproductibilité et la traçabilité,
- le coût nul en infrastructure cloud pour le prototype.

## 6. Évaluation
### 6.1 Métriques principales
- Accuracy : 0.75
- ROC-AUC : 0.8811
- F1-score global : 0.6033

### 6.2 Performances par classe
- Classe `non sous-desservie` : précision 0.99, rappel 0.70, F1-score 0.82.
- Classe `sous-desservie` : précision 0.44, rappel 0.97, F1-score 0.60.

### 6.3 Interprétation
Le modèle est très performant pour détecter les lignes sous-desservies (rappel élevé), ce qui est un point critique pour ce cas d’usage. La précision plus faible sur cette classe indique une tendance à produire des faux positifs, mais ce compromis est acceptable lorsque l’enjeu est de ne pas manquer de vraies sous-dessertes.

## 7. API REST
### 7.1 Endpoints disponibles
- `GET /health` : vérifie que l’API est disponible.
- `POST /predict` : prédit le statut d’une liaison ferroviaire.

### 7.2 Données d’entrée attendues
L’endpoint `/predict` accepte les champs suivants :
- `departure_country` : code ISO-2 du pays de départ,
- `service_type` : `day` ou `night`,
- `distance_km` : distance en kilomètres,
- `co2_per_pkm` : émissions de CO2 par passager-km,
- `arrival_country` : code ISO-2 du pays d’arrivée (optionnel).

### 7.3 Réponse
La réponse renvoyée contient :
- `is_underserved` : 0 ou 1,
- `probability` : probabilité estimée de sous-desserte,
- `label` : `sous-desservie` ou `desservie`,
- `inputs` : les valeurs de prédiction utilisées.

## 8. Déploiement
L’API peut être lancée localement avec la commande :

```bash
uvicorn api.main:app --reload
```

La documentation interactive Swagger est disponible par défaut sur `http://127.0.0.1:8000/docs`.

## 9. Limites identifiées
- Les données utilisées sont partiellement simulées ou issues de sources hétérogènes.
- Le déséquilibre des classes reste important.
- Le modèle favorise le rappel sur la classe `sous-desservie`, ce qui entraîne des faux positifs.
- La validation croisée et les tests devraient être approfondis sur de nouveaux jeux de données réels.

## 10. Perspectives d’amélioration
- Réaliser un enrichissement du jeu de données avec des données réelles de fréquentation et d’offre.
- Compléter l’analyse d’explicabilité avec SHAP/LIME pour chaque prédiction.
- Ajouter un module de surveillance de performance en production (drift, dérive des features).
- Envisager un déploiement cloud sur Azure Machine Learning ou une solution similaire si le volume de données augmente.
- Proposer un tableau de bord métier pour visualiser les lignes sous-desservies identifiées.

## 11. Conclusion
ObRail propose une démonstration complète d’un outil de détection des sous-dessertes ferroviaires. Le pipeline construit est opérationnel, la prédiction est exposée via une API REST et les résultats montrent une capacité solide à identifier les lignes réellement sous-desservies. Le travail suivant consiste à consolider les données, améliorer la robustesse du modèle et formaliser un déploiement production.
