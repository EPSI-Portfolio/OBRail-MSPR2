"""
train_rf_xgb.py — ObRail MSPR 2025-2026
Auteur : Jeannette
Rôle   : Entraînement Random Forest et XGBoost
         Tâche : classification is_flight_competitive (train compétitif vs avion ?)
         → produit models/rf_model.joblib et models/xgb_model.joblib
         → produit evaluation/comparison_rf_xgb.csv
"""

import pandas as pd
import numpy as np
import os
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

PROCESSED_DIR  = "data/processed"
MODELS_DIR     = "models"
EVAL_DIR       = "evaluation"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 — Chargement des données")
print("=" * 60)

df = pd.read_csv(f"{PROCESSED_DIR}/routes_processed.csv")
print(f"✅ routes_processed.csv chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ─────────────────────────────────────────────
# 2. PRÉPARATION DES FEATURES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 — Préparation des features")
print("=" * 60)

# Features numériques directement utilisables
NUMERIC_FEATURES = [
    'distance_km',
    'train_gco2_pkm',
    'plane_gco2_pkm',
    'train_co2_kg',
    'plane_co2_kg',
    'co2_savings_kg',
    'savings_percent',
    'co2_ratio_train_plane',
    'is_cross_border',
    'is_outlier_distance',
    'type_encoded',
]

# Features catégorielles à encoder
CATEGORICAL_FEATURES = [
    'origin_country',
    'destination_country',
    'distance_category',
    'operator',
]

TARGET = 'is_flight_competitive'

# Encodage des variables catégorielles
encoders = {}
df_model = df.copy()

for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

ENCODED_FEATURES = [col + '_enc' for col in CATEGORICAL_FEATURES]
ALL_FEATURES = NUMERIC_FEATURES + ENCODED_FEATURES

X = df_model[ALL_FEATURES]
y = df_model[TARGET]

print(f"Features utilisées ({len(ALL_FEATURES)}) :")
for f in ALL_FEATURES:
    print(f"  - {f}")
print(f"\nCible : {TARGET}")
print(f"Distribution cible :\n{y.value_counts()}")

# ─────────────────────────────────────────────
# 3. DÉCOUPAGE TRAIN / VALIDATION / TEST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 — Découpage train / validation / test")
print("=" * 60)

# 70% train / 15% validation / 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print(f"Train      : {X_train.shape[0]} lignes ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation : {X_val.shape[0]} lignes ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test       : {X_test.shape[0]} lignes ({X_test.shape[0]/len(X)*100:.1f}%)")

# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 — Entraînement Random Forest")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("✅ Random Forest entraîné")

# Évaluation sur validation
y_val_pred_rf = rf.predict(X_val)
y_val_proba_rf = rf.predict_proba(X_val)[:, 1]

rf_val_metrics = {
    'accuracy':  round(accuracy_score(y_val, y_val_pred_rf), 4),
    'f1':        round(f1_score(y_val, y_val_pred_rf, average='weighted'), 4),
    'precision': round(precision_score(y_val, y_val_pred_rf, average='weighted'), 4),
    'recall':    round(recall_score(y_val, y_val_pred_rf, average='weighted'), 4),
    'roc_auc':   round(roc_auc_score(y_val, y_val_proba_rf), 4),
}

print(f"\nMétriques Random Forest (validation) :")
for k, v in rf_val_metrics.items():
    print(f"  {k:12} : {v}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"\nCross-validation F1 (5-fold) :")
print(f"  Scores : {[round(s, 4) for s in rf_cv_scores]}")
print(f"  Moyenne : {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT XGBOOST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 — Entraînement XGBoost")
print("=" * 60)

# Calcul du ratio pour gérer le déséquilibre des classes
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    eval_metric='logloss',
    verbosity=0,
    n_jobs=-1
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print("✅ XGBoost entraîné")

# Évaluation sur validation
y_val_pred_xgb = xgb.predict(X_val)
y_val_proba_xgb = xgb.predict_proba(X_val)[:, 1]

xgb_val_metrics = {
    'accuracy':  round(accuracy_score(y_val, y_val_pred_xgb), 4),
    'f1':        round(f1_score(y_val, y_val_pred_xgb, average='weighted'), 4),
    'precision': round(precision_score(y_val, y_val_pred_xgb, average='weighted'), 4),
    'recall':    round(recall_score(y_val, y_val_pred_xgb, average='weighted'), 4),
    'roc_auc':   round(roc_auc_score(y_val, y_val_proba_xgb), 4),
}

print(f"\nMétriques XGBoost (validation) :")
for k, v in xgb_val_metrics.items():
    print(f"  {k:12} : {v}")

# Cross-validation
xgb_cv_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"\nCross-validation F1 (5-fold) :")
print(f"  Scores : {[round(s, 4) for s in xgb_cv_scores]}")
print(f"  Moyenne : {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 6. ÉVALUATION FINALE SUR TEST
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Évaluation finale sur jeu de test")
print("=" * 60)

# Random Forest sur test
y_test_pred_rf = rf.predict(X_test)
y_test_proba_rf = rf.predict_proba(X_test)[:, 1]

rf_test_metrics = {
    'model':     'Random Forest',
    'accuracy':  round(accuracy_score(y_test, y_test_pred_rf), 4),
    'f1':        round(f1_score(y_test, y_test_pred_rf, average='weighted'), 4),
    'precision': round(precision_score(y_test, y_test_pred_rf, average='weighted'), 4),
    'recall':    round(recall_score(y_test, y_test_pred_rf, average='weighted'), 4),
    'roc_auc':   round(roc_auc_score(y_test, y_test_proba_rf), 4),
    'cv_f1_mean': round(rf_cv_scores.mean(), 4),
    'cv_f1_std':  round(rf_cv_scores.std(), 4),
}

# XGBoost sur test
y_test_pred_xgb = xgb.predict(X_test)
y_test_proba_xgb = xgb.predict_proba(X_test)[:, 1]

xgb_test_metrics = {
    'model':     'XGBoost',
    'accuracy':  round(accuracy_score(y_test, y_test_pred_xgb), 4),
    'f1':        round(f1_score(y_test, y_test_pred_xgb, average='weighted'), 4),
    'precision': round(precision_score(y_test, y_test_pred_xgb, average='weighted'), 4),
    'recall':    round(recall_score(y_test, y_test_pred_xgb, average='weighted'), 4),
    'roc_auc':   round(roc_auc_score(y_test, y_test_proba_xgb), 4),
    'cv_f1_mean': round(xgb_cv_scores.mean(), 4),
    'cv_f1_std':  round(xgb_cv_scores.std(), 4),
}

print(f"\n{'Métrique':<15} {'Random Forest':>15} {'XGBoost':>15}")
print("-" * 47)
for key in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'cv_f1_mean', 'cv_f1_std']:
    print(f"{key:<15} {rf_test_metrics[key]:>15} {xgb_test_metrics[key]:>15}")

# Rapport détaillé
print(f"\n--- Rapport Random Forest ---")
print(classification_report(y_test, y_test_pred_rf))

print(f"\n--- Rapport XGBoost ---")
print(classification_report(y_test, y_test_pred_xgb))

# ─────────────────────────────────────────────
# 7. IMPORTANCE DES FEATURES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 7 — Importance des features")
print("=" * 60)

rf_importances = pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
xgb_importances = pd.Series(xgb.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)

print(f"\nTop 10 features — Random Forest :")
print(rf_importances.head(10).to_string())

print(f"\nTop 10 features — XGBoost :")
print(xgb_importances.head(10).to_string())

# ─────────────────────────────────────────────
# 8. SAUVEGARDE DES MODÈLES ET RÉSULTATS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 8 — Sauvegarde")
print("=" * 60)

# Sauvegarder les modèles
joblib.dump(rf,  f"{MODELS_DIR}/rf_model.joblib")
joblib.dump(xgb, f"{MODELS_DIR}/xgb_model.joblib")
joblib.dump(encoders, f"{MODELS_DIR}/label_encoders.joblib")
print(f"✅ rf_model.joblib sauvegardé")
print(f"✅ xgb_model.joblib sauvegardé")
print(f"✅ label_encoders.joblib sauvegardé")

# Tableau comparatif pour Louis (EDA/visualisations)
comparison_df = pd.DataFrame([rf_test_metrics, xgb_test_metrics])
comparison_df.to_csv(f"{EVAL_DIR}/comparison_rf_xgb.csv", index=False)
print(f"✅ comparison_rf_xgb.csv sauvegardé → {EVAL_DIR}/")

# Sauvegarder les importances
importances_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'rf_importance': rf.feature_importances_,
    'xgb_importance': xgb.feature_importances_,
})
importances_df.to_csv(f"{EVAL_DIR}/feature_importances_rf_xgb.csv", index=False)
print(f"✅ feature_importances_rf_xgb.csv sauvegardé → {EVAL_DIR}/")

# Sauvegarder les features utilisées (pour Charlotte et l'API)
with open(f"{MODELS_DIR}/features_rf_xgb.json", "w") as f:
    json.dump(ALL_FEATURES, f, indent=2)
print(f"✅ features_rf_xgb.json sauvegardé → {MODELS_DIR}/")

print(f"\n🎉 train_rf_xgb.py terminé avec succès !")
print(f"   Livrables prêts pour Louis (comparison_rf_xgb.csv) et Charlotte (modèles de base)")