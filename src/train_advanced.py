"""
train_advanced.py — ObRail MSPR 2025-2026
Auteure : Charlotte

Rôle :
Entraînement et comparaison des 4 modèles candidats sur la tâche
de classification binaire is_underserved. Version script du notebook
03_models.

Entrées : data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
Sorties : models/best_model.joblib
          models/model_metadata.json
          evaluation/model_comparison.csv

Usage :
    python src/train_advanced.py
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


SEED = 42
np.random.seed(SEED)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EVAL_DIR = Path("evaluation")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("ÉTAPE 1 — Chargement des splits")
    print("=" * 60)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    print(f"✅ X_train : {X_train.shape}  |  X_test : {X_test.shape}")

    # Ratio pour gérer le déséquilibre dans LightGBM
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"   scale_pos_weight : {scale_pos_weight:.2f}")

    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Définition des modèles")
    print("=" * 60)

    # n_jobs=1 pour éviter les blocages selon l'environnement
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=SEED, n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=scale_pos_weight, n_estimators=200,
            random_state=SEED, n_jobs=1, verbose=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=300, random_state=SEED,
            early_stopping=True, validation_fraction=0.1,
        ),
    }
    print(f"✅ {len(models)} modèles définis")

    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Entraînement et évaluation")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")

        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        print(f"CV F1      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        print(f"CV ROC-AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        test_f1 = f1_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)
        print(f"Test F1    : {test_f1:.4f}")
        print(f"Test AUC   : {test_auc:.4f}")

        results[name] = {
            "cv_f1_mean": round(cv_f1.mean(), 4),
            "cv_f1_std": round(cv_f1.std(), 4),
            "cv_auc_mean": round(cv_auc.mean(), 4),
            "test_f1": round(test_f1, 4),
            "test_auc": round(test_auc, 4),
            "model": model,
        }

    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Tableau comparatif")
    print("=" * 60)

    comparison = pd.DataFrame([
        {
            "Modèle": name,
            "CV F1": r["cv_f1_mean"],
            "CV F1 std": r["cv_f1_std"],
            "CV AUC": r["cv_auc_mean"],
            "Test F1": r["test_f1"],
            "Test AUC": r["test_auc"],
        }
        for name, r in results.items()
    ]).sort_values("Test F1", ascending=False).reset_index(drop=True)

    print(comparison.to_string(index=False))
    comparison.to_csv(EVAL_DIR / "model_comparison.csv", index=False)
    print(f"\n✅ Tableau sauvegardé → evaluation/model_comparison.csv")

    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Sélection et sauvegarde du meilleur modèle")
    print("=" * 60)

    best_name = max(results, key=lambda n: results[n]["test_f1"])
    best_model = results[best_name]["model"]
    best_f1 = results[best_name]["test_f1"]
    best_auc = results[best_name]["test_auc"]

    print(f"✅ Modèle sélectionné : {best_name} (F1={best_f1}, AUC={best_auc})")

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    metadata = {
        "model_name": best_name,
        "test_f1": best_f1,
        "test_roc_auc": best_auc,
        "features": list(X_train.columns),
        "target": "is_underserved",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "seed": SEED,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✅ best_model.joblib + model_metadata.json sauvegardés")
    print("\n🎉 train_advanced.py terminé avec succès !")


if __name__ == "__main__":
    main()