"""
data_prep.py — ObRail MSPR 2025-2026
Auteur : Jeannette
Rôle   : Nettoyage, traitement des valeurs manquantes et fusion des données
         → produit data/processed/routes_processed.csv
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
RAW_DIR       = "data/raw"
EXTERNAL_DIR  = "data/external"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 1 — Chargement des données")
print("=" * 60)

df = pd.read_csv(f"{RAW_DIR}/environmental_impact.csv")
airports = pd.read_csv(
    f"{EXTERNAL_DIR}/openflights_airports.csv",
    header=None,
    names=[
        "airport_id", "name", "city", "country",
        "iata", "icao", "lat", "lon",
        "altitude", "timezone", "dst", "tz_db",
        "type", "source"
    ]
)

print(f"✅ environmental_impact.csv chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"✅ openflights_airports.csv chargé : {airports.shape[0]} aéroports")

# ─────────────────────────────────────────────
# 2. DIAGNOSTIC INITIAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 2 — Diagnostic initial")
print("=" * 60)

print(f"\nValeurs manquantes par colonne :")
print(df.isnull().sum()[df.isnull().sum() > 0])

print(f"\nDoublons exacts : {df.duplicated().sum()}")
print(f"Doublons sur route_name : {df.duplicated(subset=['route_name']).sum()}")

print(f"\nDistribution type (day/night) :")
print(df['type'].value_counts())

print(f"\nDistribution par pays d'origine (top 10) :")
print(df['origin_country'].value_counts().head(10))

# ─────────────────────────────────────────────
# 3. TRAITEMENT DES VALEURS MANQUANTES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 3 — Traitement des valeurs manquantes")
print("=" * 60)

n_missing = df['distance_km'].isnull().sum()
print(f"\n{n_missing} distances manquantes à traiter")

# Stratégie : imputation par médiane selon la paire (origin_country, destination_country, type)
# Les noms de gares ne sont pas des codes IATA → OpenFlights ne peut pas matcher directement

def impute_distance(row, medians):
    """Impute la distance manquante par médiane de groupe, avec fallback progressif."""
    key1 = (row['origin_country'], row['destination_country'], row['type'])
    key2 = (row['origin_country'], row['destination_country'])
    key3 = row['origin_country']

    if key1 in medians['level1']:
        return medians['level1'][key1]
    elif key2 in medians['level2']:
        return medians['level2'][key2]
    elif key3 in medians['level3']:
        return medians['level3'][key3]
    else:
        return medians['global']

# Calculer les médianes sur les lignes NON manquantes
df_known = df[df['distance_km'].notna()]

medians = {
    'level1': df_known.groupby(['origin_country', 'destination_country', 'type'])['distance_km'].median().to_dict(),
    'level2': df_known.groupby(['origin_country', 'destination_country'])['distance_km'].median().to_dict(),
    'level3': df_known.groupby('origin_country')['distance_km'].median().to_dict(),
    'global': df_known['distance_km'].median()
}

mask_missing = df['distance_km'].isnull()
df.loc[mask_missing, 'distance_km'] = df[mask_missing].apply(
    lambda row: impute_distance(row, medians), axis=1
)

# Recalculer les colonnes CO2 dépendantes de distance_km
mask_co2_missing = df['train_co2_kg'].isnull()
df.loc[mask_co2_missing, 'train_co2_kg']   = df.loc[mask_co2_missing, 'distance_km'] * df.loc[mask_co2_missing, 'train_gco2_pkm'] / 1000
df.loc[mask_co2_missing, 'plane_co2_kg']   = df.loc[mask_co2_missing, 'distance_km'] * df.loc[mask_co2_missing, 'plane_gco2_pkm'] / 1000
df.loc[mask_co2_missing, 'co2_savings_kg'] = df.loc[mask_co2_missing, 'plane_co2_kg'] - df.loc[mask_co2_missing, 'train_co2_kg']
df.loc[mask_co2_missing, 'savings_percent'] = (
    df.loc[mask_co2_missing, 'co2_savings_kg'] / df.loc[mask_co2_missing, 'plane_co2_kg'] * 100
).round(2)

print(f"✅ {n_missing} distances imputées par médiane de groupe")
print(f"   Valeurs manquantes restantes : {df['distance_km'].isnull().sum()}")

# ─────────────────────────────────────────────
# 4. TRAITEMENT DES VALEURS ABERRANTES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 4 — Traitement des valeurs aberrantes")
print("=" * 60)

q1 = df['distance_km'].quantile(0.25)
q3 = df['distance_km'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers_mask = (df['distance_km'] < lower) | (df['distance_km'] > upper)
n_outliers = outliers_mask.sum()

print(f"\nSeuils IQR : [{lower:.1f} km — {upper:.1f} km]")
print(f"Valeurs aberrantes détectées : {n_outliers}")
print(f"\n⚠️  Ces lignes sont CONSERVÉES (longues distances réelles : TGV, Nightjet…)")
print(f"   Elles seront signalées avec un flag 'is_outlier_distance'")

df['is_outlier_distance'] = outliers_mask.astype(int)

# ─────────────────────────────────────────────
# 5. NETTOYAGE GÉNÉRAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 5 — Nettoyage général")
print("=" * 60)

# Supprimer les doublons exacts
n_before = len(df)
df = df.drop_duplicates()
print(f"✅ Doublons exacts supprimés : {n_before - len(df)}")

# Normaliser les chaînes de caractères
for col in ['origin', 'destination', 'origin_country', 'destination_country',
            'service_type', 'operator', 'type']:
    df[col] = df[col].str.strip()

# Convertir calculation_date en datetime
df['calculation_date'] = pd.to_datetime(df['calculation_date'], errors='coerce')

# Arrondir les colonnes numériques
for col in ['distance_km', 'train_co2_kg', 'plane_co2_kg', 'co2_savings_kg', 'savings_percent']:
    df[col] = df[col].round(3)

print(f"✅ Chaînes normalisées (strip)")
print(f"✅ calculation_date converti en datetime")
print(f"✅ Colonnes numériques arrondies à 3 décimales")

# ─────────────────────────────────────────────
# 6. FEATURE ENGINEERING DE BASE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Feature engineering de base")
print("=" * 60)

# Trajet transfrontalier
df['is_cross_border'] = (df['origin_country'] != df['destination_country']).astype(int)

# Catégorie de distance
df['distance_category'] = pd.cut(
    df['distance_km'],
    bins=[0, 150, 400, 800, float('inf')],
    labels=['court', 'moyen', 'long', 'tres_long']
)

# Compétitivité avion (distance < 700 km → le train est compétitif)
df['is_flight_competitive'] = (df['distance_km'] <= 700).astype(int)

# Ratio CO2 train/avion
df['co2_ratio_train_plane'] = (df['train_co2_kg'] / df['plane_co2_kg']).round(4)

# Encodage type : day=0, night=1
df['type_encoded'] = df['type'].map({'day': 0, 'night': 1})

print(f"✅ is_cross_border ajouté")
print(f"✅ distance_category ajouté (court/moyen/long/tres_long)")
print(f"✅ is_flight_competitive ajouté (distance ≤ 700 km)")
print(f"✅ co2_ratio_train_plane ajouté")
print(f"✅ type_encoded ajouté (day=0, night=1)")

# ─────────────────────────────────────────────
# 6b. SIMULATION DES VARIABLES MANQUANTES
# ─────────────────────────────────────────────
print(f"\n--- Simulation des variables manquantes ---")

# Capacité estimée selon le type de train
df['capacity'] = df['type_encoded'].map({0: 300, 1: 200})

# Fréquentation avec variation aléatoire
np.random.seed(SEED)
base_rate = 0.5 + 0.3 * df['is_cross_border'] + 0.1 * df['type_encoded']
noise = np.random.normal(0, 0.15, len(df))
fill_rate = (base_rate + noise).clip(0.3, 1.2)

df['passengers_estimated'] = (df['capacity'] * fill_rate).round(0).astype(int)

# Taux de remplissage
df['load_factor'] = (df['passengers_estimated'] / df['capacity']).round(4)

# Variable cible : ligne sous-desservie si load_factor > 0.85
df['is_underserved'] = (
    (df['load_factor'] > 0.85) & (df['distance_km'] > 150)
).astype(int)

print(f"✅ capacity ajouté")
print(f"✅ passengers_estimated ajouté")
print(f"✅ load_factor ajouté")
print(f"✅ is_underserved ajouté — distribution :")
print(df['is_underserved'].value_counts())

# ─────────────────────────────────────────────
# 7. VALIDATION FINALE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 7 — Validation finale")
print("=" * 60)

print(f"\nShape finale : {df.shape}")
print(f"\nValeurs manquantes restantes :")
missing_final = df.isnull().sum()[df.isnull().sum() > 0]
print(missing_final if len(missing_final) > 0 else "  Aucune ✅")

print(f"\nColonnes du dataset final :")
for col in df.columns:
    print(f"  - {col} ({df[col].dtype})")

print(f"\nDistribution finale type :")
print(df['type'].value_counts())

print(f"\nDistribution distance_category :")
print(df['distance_category'].value_counts())

print(f"\nTrajets transfrontaliers : {df['is_cross_border'].sum()} / {len(df)}")
print(f"Trajets compétitifs vs avion : {df['is_flight_competitive'].sum()} / {len(df)}")

# ─────────────────────────────────────────────
# 8. SAUVEGARDE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 8 — Sauvegarde")
print("=" * 60)

output_path = f"{PROCESSED_DIR}/routes_processed.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ routes_processed.csv sauvegardé → {output_path}")
print(f"   {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n🎉 data_prep.py terminé avec succès !")
print(f"   Livrable prêt pour Charlotte (feature engineering) et Louis (EDA)")