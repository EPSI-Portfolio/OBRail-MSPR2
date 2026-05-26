"""
train_rf_xgb.py — ObRail MSPR
Auteur : Jeannette
Rôle   : Détection des sous-dessertes (classification)
"""

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

import xgboost as xgb
import joblib

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
DATA_PATH = "data/processed/routes_processed.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded: {df.shape}")

# ─────────────────────────────────────────────
# 2. FEATURES / TARGET
# ─────────────────────────────────────────────
print("\nSTEP 2 — Feature selection")

features = [
    'distance_km',
    'type_encoded',
    'is_cross_border',
    'passengers_estimated',
    'capacity',
    'load_factor'
]

target = 'is_underserved'

X = df[features]
y = df[target]

print("✅ Features:", features)
print("✅ Target:", target)

print("\n📊 Distribution originale :")
print(y.value_counts(normalize=True))

# ─────────────────────────────────────────────
# 3. RÉÉQUILIBRAGE (OVERSAMPLING)
# ─────────────────────────────────────────────
print("\nSTEP 3 — Rééquilibrage")

df_majority = df[df[target] == 0]
df_minority = df[df[target] == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=SEED
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Mélanger les données
df_balanced = df_balanced.sample(frac=1, random_state=SEED)

# ✅ IMPORTANT : recréer X et y
X = df_balanced[features]
y = df_balanced[target]

print("\n📊 Nouvelle distribution (après équilibrage) :")
print(y.value_counts(normalize=True))

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print("\nSTEP 4 — Train/Test split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

# ─────────────────────────────────────────────
# 5. RANDOM FOREST
# ─────────────────────────────────────────────
print("\nSTEP 5 — Random Forest (classification)")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=SEED,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)

print(f"✅ RF Accuracy : {rf_acc:.3f}")
print("\nClassification report RF:")
print(classification_report(y_test, rf_preds))

# ─────────────────────────────────────────────
# 6. XGBOOST
# ─────────────────────────────────────────────
print("\nSTEP 6 — XGBoost (classification)")

# Calcul du poids de classe
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos if pos > 0 else 1

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=SEED,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_preds)

print(f"✅ XGB Accuracy : {xgb_acc:.3f}")
print("\nClassification report XGB:")
print(classification_report(y_test, xgb_preds))

# ─────────────────────────────────────────────
# 7. MODEL COMPARISON
# ─────────────────────────────────────────────
print("\nSTEP 7 — Model comparison")

if xgb_acc > rf_acc:
    best_model = xgb_model
    best_name = "xgboost"
else:
    best_model = rf_model
    best_name = "random_forest"

print(f"🏆 Best model: {best_name}")

# ─────────────────────────────────────────────
# 8. SAVE MODEL
# ─────────────────────────────────────────────
print("\nSTEP 8 — Saving model")

model_path = f"{MODEL_DIR}/model_underserved.joblib"

joblib.dump(best_model, model_path)

print(f"✅ Model saved at {model_path}")

# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\nSTEP 9 — Feature importance")

importances = best_model.feature_importances_

for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")

print("\n🎉 Training completed!")