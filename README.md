# 🚆 ObRail — Détection des sous-dessertes ferroviaires

## 📌 Description

ObRail est un projet de data science visant à identifier les lignes ferroviaires sous-desservies, c’est-à-dire celles où la demande en transport dépasse l’offre disponible.

Le projet repose sur :
- l’analyse de données
- le machine learning
- une API REST pour exploiter le modèle

---

## 🧠 Méthodologie

- Nettoyage et transformation des données  
- Feature engineering (distance, fréquentation, capacité, etc.)  
- Création d’une variable cible `is_underserved`  
- Entraînement de modèles de classification (RandomForest, XGBoost)  
- Rééquilibrage des données (oversampling)  

---

## 📊 Résultats

- Accuracy ≈ 0.88  
- Bonne détection des sous-dessertes  
- Modèle réaliste et interprétable  

---

## 🚀 API

L’API permet de prédire si une ligne est sous-desservie.

### 🔹 Exemple `/predict`

```json
{
  "distance_km": 800,
  "type_encoded": 0,
  "is_cross_border": 1,
  "passengers_estimated": 480,
  "capacity": 500,
  "load_factor": 0.95
}

# La reponse
{
  "is_underserved": 1
}

▶️ Lancer l’API

uvicorn api.main:app --reload


 Accès à la documentation :

http://127.0.0.1:8000/docs


📦 Installation

git clone <repo>
cd ObRail-MSPR2
pip install -r requirements.txt


📁 Structure du projet

ObRail-MSPR2/
│
├── data/
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



⚖️ Limites

Données partiellement simulées
Déséquilibre des classes
Difficulté à détecter certains cas rares


✅ Perspectives

Intégration de données réelles
Amélioration du modèle
Déploiement cloud (Azure, AWS)
Ajout de visualisations