# Dossier `src/` — Pipeline ObRail MSPR

Ce dossier contient les scripts du pipeline de machine learning, à exécuter **dans l'ordre ci-dessous**.

Chaque script lit les fichiers produits par le précédent, donc l'ordre est important. Tous les scripts se lancent depuis la **racine du projet** (pas depuis `src/`).

---

## Ordre d'exécution

### 1. `data_prep.py`
Nettoie le dataset brut et construit la cible `is_underserved`.

```bash
python src/data_prep.py
```

- **Entrée** : `data/raw/routes_europe_filtered.csv`
- **Sortie** : `data/processed/routes_processed.csv` (25 200 routes)

---

### 2. `features.py`
Construit le feature set final, encode les pays, normalise et découpe en train/test.

```bash
python src/features.py
```

- **Entrée** : `data/processed/routes_processed.csv`
- **Sorties** : `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `scaler.joblib`, `variables_retenues.csv`

---

### 3. `train_advanced.py`
Entraîne et compare les 4 modèles candidats (Logistic Regression, Random Forest, LightGBM, MLP).

```bash
python src/train_advanced.py
```

- **Entrées** : les splits produits par `features.py`
- **Sorties** : `models/best_model.joblib`, `models/model_metadata.json`, `evaluation/model_comparison.csv`

---

### 4. `optimize.py`
Optimise les hyperparamètres de LightGBM (RandomizedSearch puis GridSearch).

```bash
python src/optimize.py
```

- **Entrées** : les splits produits par `features.py`
- **Sorties** : `models/best_model_optimized.joblib`, `evaluation/optimization_comparison.csv`

Attention : ce script est long (plusieurs minutes) à cause de la recherche d'hyperparamètres. Ne pas l'interrompre.

---

### 5. `evaluate.py`
Évalue le modèle final et génère les visualisations (matrice de confusion, ROC, précision-rappel).

```bash
python src/evaluate.py
```

- **Entrée** : `models/best_model_optimized.joblib`
- **Sorties** : plots dans `evaluation/plots/`, `evaluation/evaluation_report.txt`

---

### 6. `predict.py`
Pipeline de prédiction. Teste le modèle sur des exemples et expose `predict_route()` pour l'API.

```bash
python src/predict.py
```

- **Entrées** : `models/best_model_optimized.joblib`, `data/processed/scaler.joblib`
- **Sortie** : affiche 4 prédictions de test

C'est la fonction `predict_route()` de ce fichier que l'API REST importe pour la route `/predict`.

---

## Notes importantes

- **Toujours lancer depuis la racine du projet**, jamais depuis `src/`.
- Avant de commencer : activer l'environnement puis installer les dépendances —
  `source venv/bin/activate` puis `pip install -r requirements.txt`.
- Les notebooks dans `notebooks/` font le même travail que ces scripts, avec explications et visualisations. Les scripts sont la version propre et reproductible.

### Évaluation sur un environnement vierge

Les fichiers générés (`data/processed/*.csv`, `models/*.joblib`) sont dans `.gitignore` et ne sont donc pas sur GitHub. Pour permettre une évaluation **sans relancer tout le pipeline**, deux possibilités :

1. **Modèle fourni** : inclure `models/best_model_optimized.joblib` et `data/processed/scaler.joblib` dans le rendu — l'API et `predict.py` fonctionnent alors directement.
2. **Régénération** : si seul `data/raw/routes_europe_filtered.csv` est fourni, exécuter les étapes 1 à 4 pour reconstruire les données et le modèle, puis lancer `evaluate.py` / `predict.py`.

Vérifier avant le rendu que l'une de ces deux conditions est remplie (le fichier brut `data/raw/` présent, **ou** les `.joblib` du modèle inclus).