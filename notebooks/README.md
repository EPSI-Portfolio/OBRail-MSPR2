#  Notebooks — ObRail (détection des routes ferroviaires sous-desservies)

Ce dossier contient le pipeline d'analyse du projet ObRail, du jeu de données harmonisé jusqu'au modèle évalué et explicable.

**Objectif métier** : classer chaque relation ferroviaire intercité européenne comme **sous-desservie** ou non (`is_underserved`), afin d'aider ObRail Europe à identifier les zones fragiles du réseau et à orienter ses recommandations auprès des décideurs et opérateurs.

La démarche suit un workflow standard de data science : exploration → préparation → modèles candidats → optimisation → explicabilité → évaluation, complété par des analyses descriptives et une exploration d'ensemble.

---

##  Organisation du dépôt

```
data/
  raw/                  données sources
  processed/            données préparées (ml_dataset.csv, X_train/X_test, y_train/y_test, scaler.joblib…)
models/                 modèles sauvegardés (best_model_optimized.joblib = modèle retenu) + métadonnées
evaluation/
  plots/                figures produites par les notebooks
ethics/                 bias_analysis.md, limitations.md, rgpd_compliance.md
src/                    code réutilisable + predict.py (pipeline de prédiction)
notebooks/              ce dossier
```

---

##  Pipeline des notebooks

**Un notebook = un·e propriétaire** pour limiter les conflits de fusion. Les notebooks s'exécutent dans l'ordre, chacun lisant les sorties du précédent.

### Pipeline principal

| Notebook | Contenu |
|---|---|
| `01_eda.ipynb` | Analyse exploratoire : qualité des données, distribution de la cible, relations features / cible |
| `02_feature_engineering.ipynb` | Construction et sélection des features, encodage, normalisation |
| `03_models.ipynb` | Modèles candidats : régression logistique, forêt aléatoire, LightGBM, MLP |
| `04_optimization.ipynb` | Recherche d'hyperparamètres (RandomizedSearchCV, CV stratifiée 5-fold) — **LightGBM retenu** comme modèle de production |
| `05_explainability.ipynb` | Explicabilité du modèle retenu : **SHAP** (importance globale) et **LIME** (explications de prédictions individuelles) |
| `06_evaluation.ipynb` | Évaluation finale : métriques, matrice de confusion, courbes ROC |

### Notebooks complémentaires

| Notebook | Contenu |
|---|---|
| `07_cartographie.ipynb` | Carte choroplèthe du taux de sous-desserte par pays |
| `08_voting.ipynb` | VotingClassifier — assemblage des modèles (exploratoire) |
| `09_coherence_cible.ipynb` | Tests de cohérence statistique de la variable cible `is_underserved` |

---

##  Lancer le projet

Depuis la racine du dépôt :

```bash
pip install -r requirements.txt
jupyter lab        # ou : jupyter notebook
```

Ouvrir les notebooks dans l'ordre (`01` → `06`) et exécuter **Restart & Run All** pour régénérer toutes les sorties et figures.

Le modèle final est exposé via le script de prédiction, qui simule l'intégration API :

```bash
python src/predict.py
```

---

##  Reproductibilité

- **Chemins relatifs** : les notebooks lisent les données dans `../data/...` et écrivent les figures dans `../evaluation/plots/`.
- **`random_state = 42`** partout, pour des résultats reproductibles.
- **Modèle sauvegardé** : `models/best_model_optimized.joblib`, chargé directement par `05`, `06` et `src/predict.py`.

---

##  `nbstripout` (à installer par toute l'équipe)

Un notebook `.ipynb` est un fichier JSON qui contient le code **et** toutes les sorties (texte, images encodées). Lorsque deux personnes modifient le même notebook, git tente de fusionner ce JSON et peut le corrompre. `nbstripout` retire automatiquement les **sorties** au moment du commit : les notebooks versionnés ne contiennent plus que **code + markdown**, ce qui rend les diffs lisibles et évite les conflits.

Installation (une seule fois par membre, dans le dépôt) :

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

Le `.gitattributes` est déjà versionné, mais chaque membre doit lancer ces deux commandes : la définition du filtre vit dans `.git/config` (local, non partagé).

Vérifier que le filtre est actif :

```bash
git config --get filter.nbstripout.clean   # doit renvoyer une commande nbstripout
```

> Conséquence : les notebooks versionnés n'affichent plus les figures tant qu'on n'a pas fait *Restart & Run All*. Les figures livrables restent disponibles dans `evaluation/plots/`.

---

##  Règles de collaboration

1. **`git pull` avant** d'ouvrir ou d'éditer un notebook.
2. **Une branche dédiée par tâche**, puis Pull Request.
3. **N'éditer que son propre notebook.** Pour un notebook co-détenu, se coordonner avant de pousser.
4. **Ne jamais résoudre un conflit `.ipynb` à la main** dans un éditeur de texte : garder une version, régénérer l'autre.