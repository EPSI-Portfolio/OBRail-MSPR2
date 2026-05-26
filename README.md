# 🚆 ObRail — Détection des sous-dessertes ferroviaires

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/FastAPI-API-green?logo=fastapi">
  <img src="https://img.shields.io/badge/Machine%20Learning-RandomForest-orange">
  <img src="https://img.shields.io/badge/XGBoost-Classification-red">
</p>

---

## 📌 Description

**ObRail** est un projet de Data Science visant à identifier les lignes ferroviaires **sous-desservies**, c’est-à-dire les lignes où :

> 🚉 la demande en transport dépasse l’offre disponible.

Le projet combine :

- 📊 Analyse de données
- 🤖 Machine Learning
- ⚡ API REST avec FastAPI

---

## 🧠 Méthodologie

### Étapes du projet

- Nettoyage et transformation des données
- Feature engineering :
  - distance
  - fréquentation
  - capacité
  - taux de remplissage
- Création de la variable cible `is_underserved`
- Entraînement de modèles :
  - Random Forest
  - XGBoost
- Rééquilibrage des données (oversampling)

---

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Accuracy | **0.88** |
| Détection des sous-dessertes | ✅ Bonne |
| Interprétabilité | ✅ Élevée |

---

## 🚀 API

L’API permet de prédire si une ligne ferroviaire est sous-desservie.

### 🔹 Endpoint

```http
POST /predict
```

### 🔹 Exemple de requête

```json
{
  "distance_km": 800,
  "type_encoded": 0,
  "is_cross_border": 1,
  "passengers_estimated": 480,
  "capacity": 500,
  "load_factor": 0.95
}
```

### 🔹 Exemple de réponse

```json
{
  "is_underserved": 1
}
```

---

## ▶️ Lancer l’API

```bash
uvicorn api.main:app --reload
```

### Documentation Swagger

```txt
http://127.0.0.1:8000/docs
```

---

## 📦 Installation

```bash
git clone <repo>
cd ObRail-MSPR2
pip install -r requirements.txt
```

---

## 📁 Structure du projet

```txt
ObRail-MSPR2/
│
├── data/
│
├── src/
│   ├── data_prep.py
│   └── train_rf_xgb.py
│
├── models/
│
├── api/
│   ├── routes/
│   ├── schemas.py
│   ├── model_loader.py
│   └── main.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚖️ Limites

- Données partiellement simulées
- Déséquilibre des classes
- Difficulté à détecter certains cas rares

---

## ✅ Perspectives d’amélioration

- Intégration de données réelles
- Amélioration des performances du modèle
- Déploiement cloud (Azure / AWS)
- Ajout de visualisations interactives

---

## 🛠️ Technologies utilisées

- Python
- Pandas
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre d’un projet MSPR / Data Science.