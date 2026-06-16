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
- **Sorties** : `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `scaler.joblib`, `variables_retenues.csv`

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

⚠️ Ce script est long (plusieurs minutes) à cause de la recherche d'hyperparamètres. Ne pas l'interrompre.

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

- **Entrée** : `models/best_model_optimized.joblib`, `data/processed/scaler.joblib`
- **Sortie** : affiche 4 prédictions de test

C'est la fonction `predict_route()` de ce fichier que l'API REST importe pour la route `/predict`.

---

## Notes importantes

- **Toujours lancer depuis la racine du projet**, jamais depuis `src/`.
- Les fichiers `data/processed/*.csv` et `models/*.joblib` ne sont **pas** sur GitHub (ils sont dans `.gitignore`). Il faut lancer les scripts pour les générer.
- Avant de commencer, activer l'environnement : `source venv/bin/activate` puis `pip install -r requirements.txt`.
- Les notebooks dans `notebooks/` font le même travail que ces scripts mais avec les explications et visualisations. Les scripts sont la version propre et reproductible.

---
