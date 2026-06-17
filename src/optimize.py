"""
optimize.py — ObRail MSPR 2025-2026
Auteure : Charlotte

Rôle :
Optimisation des hyperparamètres de LightGBM en deux étapes
(RandomizedSearchCV puis GridSearchCV). Version script du notebook
04_optimization.

Entrées : data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
Sorties : models/best_model_optimized.joblib
          models/model_metadata.json
          evaluation/optimization_comparison.csv

Usage :
    python src/optimize.py
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, StratifiedKFold,
)
from sklearn.metrics import f1_score, roc_auc_score, classification_report


SEED = 42
np.random.seed(SEED)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EVAL_DIR = Path("evaluation")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def neighbors(val, options):
    """Retourne val et ses voisins immédiats dans la liste options."""
    if val not in options:
        return [val]
    idx = options.index(val)
    return list(set(options[max(0, idx - 1):idx + 2]))


def main() -> None:
    print("=" * 60)
    print("ÉTAPE 1 — Chargement")
    print("=" * 60)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"✅ X_train : {X_train.shape}  |  scale_pos_weight : {scale_pos_weight:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Baseline (hyperparamètres par défaut)")
    print("=" * 60)

    baseline = LGBMClassifier(
        scale_pos_weight=scale_pos_weight, n_estimators=200,
        random_state=SEED, n_jobs=1, verbose=-1,
    )
    baseline.fit(X_train, y_train)
    baseline_f1 = f1_score(y_test, baseline.predict(X_test))
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
    print(f"Baseline F1 : {baseline_f1:.4f}  |  AUC : {baseline_auc:.4f}")

    print("\n" + "=" * 60)
    print("ÉTAPE 3 — RandomizedSearchCV (large exploration)")
    print("=" * 60)

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [-1, 5, 7, 10, 15],
        "num_leaves": [20, 31, 50, 70, 100],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    random_search = RandomizedSearchCV(
        LGBMClassifier(scale_pos_weight=scale_pos_weight,
                       random_state=SEED, n_jobs=1, verbose=-1),
        param_distributions=param_dist,
        n_iter=40, scoring="f1", cv=cv,
        random_state=SEED, n_jobs=1, verbose=1,
    )
    random_search.fit(X_train, y_train)
    best_random = random_search.best_estimator_
    random_f1 = f1_score(y_test, best_random.predict(X_test))
    random_auc = roc_auc_score(y_test, best_random.predict_proba(X_test)[:, 1])
    print(f"\n✅ RandomizedSearch F1 : {random_f1:.4f}  |  AUC : {random_auc:.4f}")

    print("\n" + "=" * 60)
    print("ÉTAPE 4 — GridSearchCV (affinement)")
    print("=" * 60)

    bp = random_search.best_params_
    param_grid = {
        "n_estimators": neighbors(bp["n_estimators"], [100, 200, 300, 500]),
        "learning_rate": neighbors(bp["learning_rate"], [0.01, 0.05, 0.1, 0.2]),
        "max_depth": neighbors(bp["max_depth"], [-1, 5, 7, 10, 15]),
        "num_leaves": neighbors(bp["num_leaves"], [20, 31, 50, 70, 100]),
        "min_child_samples": neighbors(bp["min_child_samples"], [10, 20, 30, 50]),
        "subsample": neighbors(bp["subsample"], [0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": neighbors(bp["colsample_bytree"], [0.6, 0.7, 0.8, 0.9, 1.0]),
    }

    grid_search = GridSearchCV(
        LGBMClassifier(scale_pos_weight=scale_pos_weight,
                       random_state=SEED, n_jobs=1, verbose=-1),
        param_grid=param_grid, scoring="f1", cv=cv, n_jobs=1, verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_grid = grid_search.best_estimator_
    grid_f1 = f1_score(y_test, best_grid.predict(X_test))
    grid_auc = roc_auc_score(y_test, best_grid.predict_proba(X_test)[:, 1])
    print(f"\n✅ GridSearch F1 : {grid_f1:.4f}  |  AUC : {grid_auc:.4f}")

    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Sélection et sauvegarde du modèle final")
    print("=" * 60)

    if grid_f1 >= random_f1:
        final_model, final_name, final_params = best_grid, "LightGBM (GridSearch)", grid_search.best_params_
        final_f1, final_auc = grid_f1, grid_auc
    else:
        final_model, final_name, final_params = best_random, "LightGBM (RandomizedSearch)", random_search.best_params_
        final_f1, final_auc = random_f1, random_auc

    print(f"✅ Modèle final : {final_name} (F1={final_f1:.4f})")
    print("\nClassification report :")
    print(classification_report(y_test, final_model.predict(X_test),
          target_names=["Non sous-desservi", "Sous-desservi"]))

    # Tableau comparatif
    comparison = pd.DataFrame([
        {"Étape": "Baseline", "Test F1": round(baseline_f1, 4), "Test AUC": round(baseline_auc, 4)},
        {"Étape": "RandomizedSearch", "Test F1": round(random_f1, 4), "Test AUC": round(random_auc, 4)},
        {"Étape": "GridSearch", "Test F1": round(grid_f1, 4), "Test AUC": round(grid_auc, 4)},
    ])
    comparison.to_csv(EVAL_DIR / "optimization_comparison.csv", index=False)
    print("✅ Tableau sauvegardé → evaluation/optimization_comparison.csv")

    # Sauvegarde du modèle
    joblib.dump(final_model, MODELS_DIR / "best_model_optimized.joblib")

    metadata = {
        "model_name": final_name,
        "test_f1": round(final_f1, 4),
        "test_roc_auc": round(final_auc, 4),
        "baseline_f1": round(baseline_f1, 4),
        "gain_f1": round(final_f1 - baseline_f1, 4),
        "best_params": final_params,
        "features": list(X_train.columns),
        "target": "is_underserved",
        "optimization": "RandomizedSearchCV (40 iter) -> GridSearchCV",
        "seed": SEED,
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✅ best_model_optimized.joblib + model_metadata.json sauvegardés")
    print(f"\n   Baseline F1 : {baseline_f1:.4f}")
    print(f"   Final F1    : {final_f1:.4f}")
    print(f"   Gain        : {final_f1 - baseline_f1:+.4f}")
    print("\n🎉 optimize.py terminé avec succès !")


if __name__ == "__main__":
    main()