"""
data_prep.py — ObRail MSPR 2025-2026
Auteure : Charlotte
Rôle    : Construction du dataset depuis les données GTFS réelles
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

RAW_DIR      = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# ÉTAPE 1 — CHARGEMENT DES DONNÉES GTFS
# ─────────────────────────────────────────────
# On charge les fichiers routes.txt et trips.txt depuis chaque
# dossier GTFS. Les données sont organisées par pays et type de service
# (jour/nuit). Pour chaque route on calcule le nombre de trips
# qui lui sont associés — c'est notre proxy de fréquence de service.

print("=" * 60)
print("ÉTAPE 1 — Chargement des données GTFS")
print("=" * 60)

def load_gtfs_routes(base_path, service_type, country):
    """Charge routes + trips depuis un dossier GTFS."""
    routes_path = f"{base_path}/routes.txt"
    trips_path  = f"{base_path}/trips.txt"

    if not os.path.exists(routes_path) or not os.path.exists(trips_path):
        return None

    routes = pd.read_csv(routes_path, low_memory=False)
    trips  = pd.read_csv(trips_path,  low_memory=False)

    trip_counts = trips.groupby('route_id').size().reset_index(name='trip_count')

    df = routes.merge(trip_counts, on='route_id', how='left')
    df['service_type'] = service_type
    df['country']      = country

    return df

day_sources = [
    ('day/Denmark',               'day',   'DK'),
    ('day/Eurostar_international', 'day',   'EU'),
    ('day/France',                 'day',   'FR'),
    ('day/Germany',                'day',   'DE'),
    ('day/Switzerland',            'day',   'CH'),
]

night_sources = [
    ('night/long_distance',       'night', 'EU'),
    ('night/long_distance_rail',  'night', 'EU'),
    ('night/open_data',           'night', 'EU'),
    ('night/sncf-data',           'night', 'FR'),
    ('night/Switzerland',         'night', 'CH'),
]

frames = []
for folder, service_type, country in day_sources + night_sources:
    path = f"{RAW_DIR}/{folder}"
    df   = load_gtfs_routes(path, service_type, country)
    if df is not None:
        frames.append(df)
        print(f"✅ {folder:40s} → {len(df):6d} routes")
    else:
        print(f"⚠️  {folder:40s} → fichiers manquants")

df = pd.concat(frames, ignore_index=True)
print(f"\n✅ Total brut chargé : {df.shape[0]:,} routes × {df.shape[1]} colonnes")

# ─────────────────────────────────────────────
# ÉTAPE 2 — NETTOYAGE ET STANDARDISATION
# ─────────────────────────────────────────────
# On garde uniquement les colonnes nécessaires, on nettoie les noms
# de routes, on supprime les doublons et on impute les trip_count
# manquants par 1 (route enregistrée mais aucun trip associé).

print("\n" + "=" * 60)
print("ÉTAPE 2 — Nettoyage et standardisation")
print("=" * 60)

cols_keep = ['route_id', 'route_short_name', 'route_long_name',
             'route_type', 'trip_count', 'service_type', 'country']
df = df[cols_keep].copy()

# Construire un nom de route propre
df['route_name'] = (
    df['route_long_name']
    .fillna(df['route_short_name'])
    .fillna('Unknown')
    .str.strip()
)

# Supprimer les routes sans nom
before = len(df)
df = df[df['route_name'] != 'Unknown'].copy()
print(f"✅ Routes sans nom supprimées : {before - len(df)}")

# Supprimer les doublons
before = len(df)
df = df.drop_duplicates(subset=['route_id', 'country', 'service_type'])
print(f"✅ Doublons supprimés        : {before - len(df)}")

# Imputer trip_count manquant
df['trip_count'] = df['trip_count'].fillna(1).astype(int)

print(f"\n✅ Shape après nettoyage : {df.shape}")
print(f"\nRépartition service_type :\n{df['service_type'].value_counts()}")
print(f"\nRépartition country :\n{df['country'].value_counts()}")

# ─────────────────────────────────────────────
# ÉTAPE 3 — FILTRAGE INTERCITY UNIQUEMENT
# ─────────────────────────────────────────────
# On filtre pour ne garder que les vrais trains intercity/longue distance.
# En GTFS, route_type 2 = Rail, 100 = Railway Service, 109 = Suburban Rail.
# On exclut la Suisse dont les données GTFS sont trop granulaires
# (elles incluent bus, trams, métros locaux) et dominent le dataset
# avec 146,000 routes sur 151,000 au total.

print("\n" + "=" * 60)
print("ÉTAPE 3 — Filtrage des routes intercity")
print("=" * 60)

before = len(df)

df = df[
    (df['route_type'].isin([2, 100, 109])) &
    (df['country'] != 'CH')
].copy()

print(f"✅ Routes non-intercity supprimées : {before - len(df):,}")
print(f"✅ Shape après filtrage            : {df.shape}")
print(f"\nRépartition finale :")
print(df.groupby(['country', 'service_type']).size())
print(f"\nTrip count stats :")
print(df['trip_count'].describe().round(1))

for country in df['country'].unique():
    print(f"\n--- {country} exemples ---")
    print(df[df['country'] == country]['route_name'].head(5).tolist())

# ─────────────────────────────────────────────
# ÉTAPE 4 — CONSTRUCTION DE LA CIBLE is_underserved
# ─────────────────────────────────────────────
# Un trajet est sous-desservi si son nombre de trips est inférieur
# au 25e percentile de son groupe (pays + service_type).
# Cette définition est basée sur des données réelles GTFS —
# pas de simulation.

print("\n" + "=" * 60)
print("ÉTAPE 4 — Construction de la cible is_underserved")
print("=" * 60)

# Calculer le 25e percentile par groupe
p25 = df.groupby(['country', 'service_type'])['trip_count'].transform(
    lambda x: x.quantile(0.25)
)

df['is_underserved'] = (df['trip_count'] <= p25).astype(int)

print("Distribution is_underserved :")
print(df['is_underserved'].value_counts())
print(df['is_underserved'].value_counts(normalize=True).mul(100).round(1))

print("\nTaux de sous-desserte par groupe :")
print(df.groupby(['country', 'service_type'])['is_underserved'].mean().mul(100).round(1))  

# ─────────────────────────────────────────────
# ÉTAPE 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
# On crée des features supplémentaires à partir des données existantes
# pour enrichir le dataset avant la modélisation.

print("\n" + "=" * 60)
print("ÉTAPE 5 — Feature engineering")
print("=" * 60)

# Encodage service_type
df['type_encoded'] = df['service_type'].map({'day': 0, 'night': 1})

# Encodage pays
df['country_encoded'] = df['country'].map({
    'DE': 0, 'DK': 1, 'EU': 2, 'FR': 3
})

# Catégorie de fréquence
df['frequency_category'] = pd.cut(
    df['trip_count'],
    bins=[0, 2, 7, 30, float('inf')],
    labels=['très_faible', 'faible', 'moyen', 'élevé']
)

# Log du trip_count pour réduire l'asymétrie
df['log_trip_count'] = np.log1p(df['trip_count'])

# Est-ce une route internationale (EU)
df['is_international'] = (df['country'] == 'EU').astype(int)

print(f"✅ Features créées")
print(f"\nfréquency_category distribution :")
print(df['frequency_category'].value_counts())
print(f"\nlog_trip_count stats :")
print(df['log_trip_count'].describe().round(2))
print(f"\nShape : {df.shape}")

# ─────────────────────────────────────────────
# ÉTAPE 6 — VALIDATION FINALE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Validation finale")
print("=" * 60)

print(f"Shape finale : {df.shape}")
print(f"\nValeurs manquantes :")
missing = df.isnull().sum()
print(missing[missing > 0] if len(missing[missing > 0]) else "✅ Aucun missing")

print(f"\nDoublons : {df.duplicated().sum()}")

print(f"\nColonnes finales :")
for col in df.columns:
    print(f"  - {col} ({df[col].dtype})")

print(f"\nDistribution cible is_underserved :")
print(df['is_underserved'].value_counts())
print(df['is_underserved'].value_counts(normalize=True).mul(100).round(1))

print(f"\nAperçu :")
print(df[['route_name', 'country', 'service_type', 
          'trip_count', 'is_underserved']].head(10))

# ─────────────────────────────────────────────
# ÉTAPE 7 — SAUVEGARDE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 7 — Sauvegarde")
print("=" * 60)

output_path = f"{PROCESSED_DIR}/routes_processed.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ routes_processed.csv sauvegardé → {output_path}")
print(f"   {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n🎉 data_prep.py terminé avec succès !")

