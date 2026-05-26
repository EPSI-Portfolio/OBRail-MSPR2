"""
data_prep.py — ObRail MSPR 2025-2026
Auteur : Jeannette
Rôle   : Nettoyage, traitement, feature engineering + sous-dessertes
         → produit data/processed/routes_processed.csv
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 — Chargement des données")
print("=" * 60)

df = pd.read_csv(f"{RAW_DIR}/environmental_impact.csv")
print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ─────────────────────────────────────────────
# 2. DIAGNOSTIC
# ─────────────────────────────────────────────
print("\nValeurs manquantes :")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ─────────────────────────────────────────────
# 3. IMPUTATION DISTANCE
# ─────────────────────────────────────────────
print("\nÉTAPE 3 — Imputation des distances")

df_known = df[df['distance_km'].notna()]

medians = {
    'level1': df_known.groupby(['origin_country', 'destination_country', 'type'])['distance_km'].median().to_dict(),
    'level2': df_known.groupby(['origin_country', 'destination_country'])['distance_km'].median().to_dict(),
    'level3': df_known.groupby('origin_country')['distance_km'].median().to_dict(),
    'global': df_known['distance_km'].median()
}

def impute_distance(row):
    key1 = (row['origin_country'], row['destination_country'], row['type'])
    key2 = (row['origin_country'], row['destination_country'])
    key3 = row['origin_country']

    if key1 in medians['level1']:
        return medians['level1'][key1]
    elif key2 in medians['level2']:
        return medians['level2'][key2]
    elif key3 in medians['level3']:
        return medians['level3'][key3]
    return medians['global']

mask_missing = df['distance_km'].isnull()
df.loc[mask_missing, 'distance_km'] = df[mask_missing].apply(impute_distance, axis=1)
print(f"✅ Distances imputées : {mask_missing.sum()}")

# ─────────────────────────────────────────────
# 4. RECALCUL CO2
# ─────────────────────────────────────────────
print("\nÉTAPE 4 — Recalcul CO2")

mask_train = df['train_co2_kg'].isnull()
mask_plane = df['plane_co2_kg'].isnull()

df.loc[mask_train, 'train_co2_kg'] = df.loc[mask_train, 'distance_km'] * df.loc[mask_train, 'train_gco2_pkm'] / 1000
df.loc[mask_plane, 'plane_co2_kg'] = df.loc[mask_plane, 'distance_km'] * df.loc[mask_plane, 'plane_gco2_pkm'] / 1000

df['co2_savings_kg'] = df['plane_co2_kg'] - df['train_co2_kg']
df['savings_percent'] = (df['co2_savings_kg'] / df['plane_co2_kg'] * 100).round(2)

print("✅ CO2 recalculé")

# ─────────────────────────────────────────────
# 5. NETTOYAGE
# ─────────────────────────────────────────────
print("\nÉTAPE 5 — Nettoyage")

df = df.drop_duplicates()

for col in ['origin', 'destination', 'origin_country', 'destination_country', 'type']:
    df[col] = df[col].str.strip()

df['calculation_date'] = pd.to_datetime(df['calculation_date'], errors='coerce')

# ─────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\nÉTAPE 6 — Feature engineering")

df['is_cross_border'] = (df['origin_country'] != df['destination_country']).astype(int)

df['distance_category'] = pd.cut(
    df['distance_km'],
    bins=[0, 150, 400, 800, float('inf')],
    labels=['court', 'moyen', 'long', 'tres_long']
)

df['type_encoded'] = df['type'].map({'day': 0, 'night': 1})

df['co2_ratio_train_plane'] = (
    df['train_co2_kg'] / df['plane_co2_kg']
).round(4)

# ─────────────────────────────────────────────
# 7. FRÉQUENTATION (SIMULATION)
# ─────────────────────────────────────────────
print("\nÉTAPE 7 — Estimation de la fréquentation")

def estimate_capacity(row):
    return 400 if row['type'] == 'night' else 500

def estimate_load_factor(row):
    base = 0.65
    if row['distance_km'] > 500:
        base += 0.1
    if row['type'] == 'night':
        base -= 0.05
    return np.clip(np.random.normal(base, 0.05), 0.4, 0.9)

df['capacity'] = df.apply(estimate_capacity, axis=1)
df['load_factor'] = df.apply(estimate_load_factor, axis=1)

df['passengers_estimated'] = (df['capacity'] * df['load_factor']).astype(int)

print("✅ fréquentation simulée ajoutée")

# ─────────────────────────────────────────────
# 8. CO2 PAR PASSAGER
# ─────────────────────────────────────────────
print("\nÉTAPE 8 — CO2 par passager")

df['train_co2_per_passenger'] = (df['train_co2_kg'] / df['passengers_estimated']).round(4)
df['plane_co2_per_passenger'] = (df['plane_co2_kg'] / df['passengers_estimated']).round(4)

print("✅ CO2/passager ajouté")

# ─────────────────────────────────────────────
# 9. SOUS-DESSERTES (IA)
# ─────────────────────────────────────────────
print("\nÉTAPE 9 — Détection des sous-dessertes")

df['service_ratio'] = df['passengers_estimated'] / df['capacity']

df['is_underserved'] = (
    (
        df['passengers_estimated'] > df['capacity'] * 0.85
    ) &
    (
        df['distance_km'] > 400
    ) &
    (
        df['is_cross_border'] == 1
    )
).astype(int)

# ajouter un peu de bruit réaliste
noise = np.random.rand(len(df)) < 0.1
df.loc[noise, 'is_underserved'] = 1 - df.loc[noise, 'is_underserved']

print("✅ is_underserved créé")
print(df['is_underserved'].value_counts())

# ─────────────────────────────────────────────
# 10. VALIDATION
# ─────────────────────────────────────────────
print("\nÉTAPE 10 — Validation")

missing = df.isnull().sum()
print(missing[missing > 0] if len(missing[missing > 0]) else "✅ Aucun missing")

print(f"\nShape final : {df.shape}")

# ─────────────────────────────────────────────
# 11. SAVE
# ─────────────────────────────────────────────
output_path = f"{PROCESSED_DIR}/routes_processed.csv"
df.to_csv(output_path, index=False)

print("\n✅ DATASET FINAL SAUVEGARDÉ")
print(f"{df.shape[0]} lignes × {df.shape[1]} colonnes")
print("\n🎉 data_prep terminé !")