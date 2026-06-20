## 12. Benchmark des services cloud d'intelligence artificielle

Conformément au cahier des charges, une étude comparative a été menée sur quatre services représentatifs du marché : deux solutions cloud généralistes (AWS SageMaker, Google Vertex AI), une solution cloud davantage orientée Python (Azure Machine Learning), et une plateforme open source (HuggingFace AutoTrain). Chacun a été évalué sur six critères : capacités techniques, prix, transparence, explicabilité, facilité d'intégration et pertinence pour des données ferroviaires tabulaires. Le tableau ci-dessous résume cette comparaison et facilite le choix de la solution la plus adaptée au projet.

### 12.1 AWS SageMaker

SageMaker couvre l'ensemble du cycle de vie ML, avec un module Autopilot dédié à la classification sur données tabulaires — directement comparable à notre cas d'usage. Son explicabilité native (SageMaker Clarify, basée sur SHAP) est un point fort. En contrepartie, sa facturation est éclatée sur plus de 15 sous-services facturés séparément (notebook, training, endpoint, stockage, transfert), et les instances `ml.*` coûtent 20 à 40 % plus cher que l'EC2 équivalent. Un déploiement réaliste (entraînement + endpoint temps réel pendant un mois) atteint facilement 250 USD, et la courbe d'apprentissage AWS (IAM, VPC, S3) reste élevée pour une équipe non native de cet écosystème.

**Score global : 3/5** — écarté pour ce projet en raison d'une complexité et d'un coût disproportionnés par rapport au volume de données (25 200 lignes) et à la durée allouée (29h de préparation).

### 12.2 Google Vertex AI (AutoML Tables)

Vertex AI AutoML Tabular est la solution la plus proche de notre besoin sur le papier : entraînement automatisé de modèles de classification sur données structurées, avec prétraitement automatique des valeurs manquantes et de l'encodage catégoriel. Vertex Explainable AI fournit des explications basées sur les valeurs de Shapley au même tarif que l'inférence standard, ce qui est cohérent avec notre propre démarche SHAP/LIME. La plateforme est reconnue Leader du Magic Quadrant Gartner 2024 pour les services IA cloud.

Le principal frein est tarifaire : l'entraînement de modèles tabulaires personnalisés est facturé environ 21,25 USD par node-heure, et les frais de déploiement d'endpoint courent même en l'absence de requête tant que le modèle reste actif. Le crédit gratuit de 300 USD sur 90 jours pour un nouveau compte aurait été consommé avant la fin du cycle complet d'optimisation réalisé dans ce projet (RandomizedSearchCV puis GridSearchCV, soit 3 240 entraînements).

**Score global : 4/5** — écarté pour ce projet pour des raisons budgétaires malgré une excellente adéquation technique.

### 12.3 Azure Machine Learning

Azure ML propose un AutoML qui sélectionne automatiquement le meilleur algorithme parmi des dizaines de candidats avec validation croisée à 5 plis par défaut — une démarche très proche de notre propre comparaison manuelle entre Random Forest, XGBoost, LightGBM et MLP. Son modèle de facturation pay-as-you-go, sans minimum imposé, est plus prévisible que celui d'AWS. L'onglet *Explanations* du studio Azure ML fournit nativement une feature importance globale et des explications par prédiction individuelle, complétées par un tableau de bord d'IA responsable (équité, détection de biais). La documentation officielle Microsoft alerte même explicitement sur le risque de data leakage (accuracy suspicieusement élevée) — un problème que notre équipe a justement rencontré et corrigé en section 7.3 du rapport.

**Score global : 4/5** — retenu comme solution cloud de référence en cas de déploiement à plus grande échelle, pour sa bonne intégration Python et son interface jugée plus intuitive que la concurrence.

### 12.4 HuggingFace AutoTrain

AutoTrain est entièrement gratuit, seule la ressource de calcul (GPU/CPU sur HuggingFace Spaces) étant facturée à la minute — c'est l'option la moins coûteuse des quatre. Son code est open source et consultable sur GitHub, garantissant une transparence totale. Cependant, le tabulaire reste une fonctionnalité secondaire de la plateforme, historiquement centrée sur le NLP et la vision : le format imposé (colonnes `id` et `target` strictes) est peu flexible face à notre feature set de 38 colonnes dont 33 issues d'un encodage one-hot, et aucun module SHAP/LIME natif n'est disponible pour le tabulaire, contrairement à l'offre NLP/vision bien plus riche en interprétabilité.

**Score global : 2/5** — écarté pour ce projet, la plateforme étant sous-dimensionnée fonctionnellement pour une classification tabulaire avancée avec explicabilité poussée.

### 12.5 Tableau de synthèse

| Service | Prix | Explicabilité | Intégration | Pertinence ferroviaire | Score |
|---|---|---|---|---|---|
| AWS SageMaker | Élevé, facturation éclatée | SHAP natif (Clarify) | Complexe (écosystème AWS) | Bonne mais surdimensionnée | 3/5 |
| Google Vertex AI | Élevé (~21,25 USD/node-h) | SHAP natif (Explainable AI) | Bonne si déjà sur GCP | Très bonne | 4/5 |
| Azure Machine Learning | Modéré, pay-as-you-go | Native (Explanations + dashboard) | Très bonne (Python natif) | Très bonne | 4/5 |
| HuggingFace AutoTrain | Très faible | Limitée sur le tabulaire | Simple mais rigide (format imposé) | Faible | 2/5 |
| **Solution interne (retenue)** | **Nul (hors temps dev.)** | **Complète (SHAP + LIME maîtrisés)** | **Totale (API FastAPI sur-mesure)** | **Optimale** | **5/5** |

### 12.6 Justification du choix d'une solution interne

Au regard de ce benchmark, l'équipe a choisi de développer et d'entraîner le modèle final localement avec des bibliothèques open source (scikit-learn, LightGBM, XGBoost) plutôt que de s'appuyer sur un service cloud managé, pour quatre raisons principales :

1. **Coût** — aucun des trois services cloud testés n'offre de mode gratuit compatible avec un cycle complet d'optimisation par hyperparamètres (3 240 entraînements réalisés dans ce projet) ; HuggingFace AutoTrain est gratuit mais inadapté techniquement.
2. **Transparence et reproductibilité** — un pipeline codé et versionné (`data_prep.py` → `features.py` → `train_advanced.py` → `optimize.py` → `evaluate.py` → `predict.py`) garantit une traçabilité complète exigée par le cahier des charges, alors que les AutoML cloud restent partiellement des boîtes noires sur leurs choix internes d'algorithmes.
3. **Maîtrise de l'explicabilité** — l'intégration manuelle de SHAP et LIME (notebook `05_explainability.ipynb`) permet un contrôle total sur l'interprétation des résultats, condition essentielle pour justifier les choix méthodologiques face au jury.
4. **Volume de données** — avec 25 200 lignes après nettoyage, le volume ne justifie pas l'infrastructure distribuée proposée par les solutions cloud, pensées pour des volumes nettement supérieurs.

Azure Machine Learning reste néanmoins identifié comme la solution la plus pertinente pour un déploiement futur à plus grande échelle (voir section 16, Perspectives), notamment pour sa bonne intégration avec l'écosystème Python déjà utilisé dans ce projet et son module d'explicabilité natif compatible avec notre approche actuelle.