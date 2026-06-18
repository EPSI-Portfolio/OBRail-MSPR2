# Veille Technique et Recommandations
**Projet** : ObRail MSPR 2025-2026 
**Date** : Juin 2026

---

## 1. Veille algorithmique — Classification tabulaire

### État de l'art
La classification sur données tabulaires reste dominée par les modèles de gradient boosting. LightGBM, XGBoost et CatBoost sont les trois références du domaine. Les réseaux de neurones profonds (MLP, TabNet, FT-Transformer) ont progressé mais ne surpassent pas systématiquement le boosting sur des datasets de taille modérée (< 100 000 lignes).

### Pourquoi LightGBM a été retenu
LightGBM a été sélectionné après comparaison de quatre modèles candidats (Régression Logistique, Random Forest, LightGBM, MLP) sur notre dataset de 25 200 routes. Il obtient le meilleur F1-score (0.605) et le meilleur ROC-AUC (0.883) sur le jeu de test. Sa rapidité d'entraînement, sa gestion native du déséquilibre de classes via `scale_pos_weight`, et ses capacités d'explicabilité via SHAP en font le choix le plus adapté aux contraintes du projet ObRail.

### Alternatives identifiées
- **XGBoost** : performances similaires à LightGBM, légèrement plus lent sur grands datasets. Aurait pu être retenu — les résultats auraient probablement été proches.
- **CatBoost** : particulièrement adapté aux variables catégorielles, aurait pu être intéressant pour l'encodage des pays. Non testé faute de temps.
- **TabNet** : réseau de neurones spécialisé pour les données tabulaires avec attention mechanism. Nécessite plus de données et de tuning. À explorer si le dataset s'agrandit.
- **Random Forest** : F1=0.582, très proche de LightGBM avec des hyperparamètres par défaut. Solution de repli fiable si LightGBM posait des problèmes d'intégration.

---

## 2. Veille sur les services IA existants

Trois grandes familles de services ont été étudiées dans le cadre du benchmark (voir `evaluation/benchmark_services_ia.csv` — travail de Louis) :

### Google Vertex AI / AutoML Tables
Permet d'entraîner des modèles sur données tabulaires sans code. Interface visuelle, déploiement automatique. Prix : facturation à l'heure de calcul et au nombre de prédictions. Avantage : intégration facile avec BigQuery. Limite : boîte noire partielle, explicabilité limitée sans surcoût.

### Azure Machine Learning Studio
Environnement MLOps complet avec pipeline automatisé, versioning des modèles, monitoring en production. Bien intégré avec l'écosystème Microsoft. Limite : complexité de configuration, coût élevé pour un projet académique.

### AWS SageMaker AutoPilot
Automatise le feature engineering, la sélection de modèle et le tuning. Génère du code Python réutilisable. Limite : dépendance à l'écosystème AWS, transparence partielle sur les choix algorithmiques.

### Pourquoi un modèle interne a été choisi
Le cahier des charges ObRail exige transparence, reproductibilité et documentation des choix algorithmiques. Les services AutoML sont des boîtes noires partielles — ils ne permettent pas de documenter précisément pourquoi un modèle prédit une route comme sous-desservie. Notre approche interne avec LightGBM + SHAP + LIME répond directement à cette exigence. De plus, le coût des services cloud sur la durée dépasse celui d'une solution interne maintenue par l'équipe data d'ObRail.

---

## 3. Risques, limites et biais identifiés

### 3.1 Dominance de log_distance (risque principal)
L'analyse SHAP révèle que `log_distance` domine largement les décisions du modèle (importance 5.17 contre 0.19 pour la feature suivante). Cela signifie que le modèle apprend principalement le seuil `distance_km > 100` de la règle de construction de la cible `is_underserved`, plutôt que des patterns opérationnels indépendants.

**Conséquence** : le modèle est performant sur ce dataset mais sa capacité de généralisation à des données réelles non simulées est limitée.

**Recommandation** : remplacer la règle `days_active <= 3 AND distance_km > 100` par une cible construite à partir de données GTFS réelles sur la fréquence hebdomadaire des dessertes. Cela réduirait la circularité entre features et cible.

### 3.2 Déséquilibre de classes
Le dataset présente un ratio 4:1 (80.6% non sous-desservi, 19.4% sous-desservi). Ce déséquilibre a été géré via `scale_pos_weight` dans LightGBM et le choix du F1-score comme métrique principale. Cependant, le modèle présente une précision de 0.44 sur la classe sous-desservie — il génère beaucoup de faux positifs (1 213 sur 5 040 routes de test).

**Conséquence** : si ObRail utilise le modèle pour prioriser des interventions terrain, environ 44% des routes flagguées nécessiteront une vérification manuelle.

**Recommandation** : ajuster le seuil de décision (actuellement 0.5) selon le coût opérationnel des faux positifs vs faux négatifs dans le contexte d'ObRail.

### 3.3 Biais géographique
Le dataset est déséquilibré géographiquement — la France (6 960 routes), l'Espagne (4 789) et l'Allemagne (4 429) représentent 64% des données. Les pays avec peu de routes (ME, MK, AL) sont sous-représentés et leurs prédictions sont moins fiables.

**Recommandation** : enrichir le dataset avec des sources GTFS complémentaires pour les pays sous-représentés.

### 3.4 Données simulées
Certaines features numériques (`co2_per_pkm`, `emissions_co2`) ont été générées par simulation. L'analyse de corrélation en EDA a montré des corrélations quasi nulles de ces features avec la cible, suggérant qu'elles n'apportent pas de signal réel. `log_co2` figure dans le feature set mais son importance SHAP est faible (0.14).

**Recommandation** : remplacer les données simulées par des données réelles issues d'ADEME ou de l'Agence Européenne de l'Environnement pour les futures itérations.

---

## 4. Cadre réglementaire et éthique

### RGPD
Le projet ne traite aucune donnée personnelle — toutes les données concernent des liaisons ferroviaires et non des individus. Les principes de minimisation et de documentation sont respectés : seules les features nécessaires à la prédiction sont conservées, et chaque choix est documenté dans les notebooks.

### EU AI Act — Ethics Guidelines for Trustworthy AI
La Commission européenne définit sept exigences pour une IA digne de confiance. Notre projet répond aux suivantes :

- **Transparence** : les décisions du modèle sont explicables via SHAP (niveau global) et LIME (niveau individuel). Chaque prédiction peut être justifiée feature par feature.
- **Robustesse technique** : validation croisée 5-fold, jeu de test séparé, seed fixée pour la reproductibilité.
- **Supervision humaine** : le modèle produit une probabilité, pas une décision finale. La décision d'intervenir sur une route reste humaine.
- **Explicabilité** : `lime_explanation.png` et les plots SHAP constituent des preuves d'audit disponibles.

**Limite identifiée** : la dominance de `log_distance` dans les décisions SHAP réduit partiellement la confiance dans l'indépendance du modèle vis-à-vis de la règle de construction de la cible. Cette limite est documentée et communiquée.

### Sécurité informatique
- Le modèle est sauvegardé en format `joblib` — format standard Python, pas de dépendance propriétaire.
- Les fichiers de données brutes ne sont pas versionnés sur GitHub (`.gitignore`).
- Le modèle ne traite pas de données sensibles — aucun risque RGPD lié à l'inférence.

---

## 5. Recommandations pour les prochaines itérations

1. **Construire une cible basée sur des données GTFS réelles** — remplacer la règle `days_active <= 3 AND distance_km > 100` par une mesure de fréquence hebdomadaire réelle issue des flux GTFS européens (Back-on-Track, Eurostat).

2. **Enrichir les features géographiques** — ajouter la densité de population des zones desservies, la présence d'aéroports concurrents, et les temps de trajet comparatifs avion/train.

3. **Ajuster le seuil de décision** — tester des seuils entre 0.3 et 0.7 selon le coût opérationnel des faux positifs pour ObRail.

4. **Mettre en place un monitoring en production** — surveiller le drift des prédictions, la distribution des probabilités retournées, et la latence de l'API `/predict`.

5. **Étendre le benchmark géographique** — enrichir les données des pays sous-représentés (Balkans, pays baltes) pour améliorer la fiabilité des prédictions dans ces régions.