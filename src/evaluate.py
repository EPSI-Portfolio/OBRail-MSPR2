"""
evaluate.py — ObRail MSPR 2025-2026

Rôle :
Évaluation finale du modèle optimisé sur le jeu de test : métriques,
matrice de confusion, courbes ROC et précision-rappel. Version script
des évaluations des notebooks 03 et 04.

Entrées : models/best_model_optimized.joblib
          data/processed/X_test.csv, y_test.csv
Sorties : evaluation/plots/confusion_matrix_final.png
          evaluation/plots/roc_curve_final.png
          evaluation/plots/precision_recall_final.png
          evaluation/evaluation_report.txt

Usage :
    python src/evaluate.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
)


SEED = 42
np.random.seed(SEED)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EVAL_DIR = Path("evaluation")
PLOT_DIR = EVAL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("ÉTAPE 1 — Chargement du modèle et du jeu de test")
    print("=" * 60)

    model = joblib.load(MODELS_DIR / "best_model_optimized.joblib")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    print(f"✅ Modèle : {type(model).__name__}")
    print(f"   X_test : {X_test.shape}")

    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Prédictions et métriques")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_f1 = f1_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(
        y_test, y_pred, target_names=["Non sous-desservi", "Sous-desservi"]
    )

    print(f"Test F1      : {test_f1:.4f}")
    print(f"Test ROC-AUC : {test_auc:.4f}")
    print("\nClassification report :")
    print(report)

    # Sauvegarde du rapport texte
    with open(EVAL_DIR / "evaluation_report.txt", "w") as f:
        f.write(f"Évaluation du modèle final — ObRail MSPR\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Test F1      : {test_f1:.4f}\n")
        f.write(f"Test ROC-AUC : {test_auc:.4f}\n\n")
        f.write(report)
    print("✅ Rapport sauvegardé → evaluation/evaluation_report.txt")

    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Matrice de confusion")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non sous-desservi", "Sous-desservi"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matrice de confusion — modèle final")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_final.png")
    plt.close()
    print("✅ confusion_matrix_final.png sauvegardée")

    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Courbe ROC")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, color="#e74c3c", linewidth=2,
            label=f"Modèle final (AUC={test_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC — modèle final")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve_final.png")
    plt.close()
    print("✅ roc_curve_final.png sauvegardée")

    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Courbe Précision-Rappel")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 6))
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ax.plot(rec, prec, color="#3498db", linewidth=2, label="Modèle final")
    ax.axhline(y_test.mean(), color="black", linestyle="--",
               label=f"Baseline (précision={y_test.mean():.2f})")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe Précision-Rappel — modèle final")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "precision_recall_final.png")
    plt.close()
    print("✅ precision_recall_final.png sauvegardée")

    print("\n🎉 evaluate.py terminé avec succès !")


if __name__ == "__main__":
    main()