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

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# ÉTAPE 1 — CHARGEMENT DES DONNÉES GTFS
# ─────────────────────────────────────────────
# On charge les fichiers routes.txt et trips.txt depuis chaque
# dossier GTFS organisé par pays et type de service (jour/nuit).
# Pour chaque route on calcule le nombre de trips associés —
# ce nombre est notre proxy de fréquence de service réelle.

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

    # Compter le nombre de trips par route = fréquence de service
    trip_counts = trips.groupby('route_id').size().reset_index(name='trip_count')

    df = routes.merge(trip_counts, on='route_id', how='left')
    df['service_type'] = service_type
    df['country']      = country

    return df

# Sources jour — trains de jour par pays
day_sources = [
    ('day/Denmark',                'day',   'DK'),
    ('day/Eurostar_international', 'day',   'EU'),
    ('day/France',                 'day',   'FR'),
    ('day/Germany',                'day',   'DE'),
    ('day/Switzerland',            'day',   'CH'),
]

# Sources nuit — trains de nuit internationaux et par pays
night_sources = [
    ('night/long_distance',        'night', 'EU'),
    ('night/long_distance_rail',   'night', 'EU'),
    ('night/open_data',            'night', 'EU'),
    ('night/sncf-data',            'night', 'FR'),
    ('night/Switzerland',          'night', 'CH'),
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
# On garde uniquement les colonnes nécessaires, on construit
# un nom de route propre, on supprime les doublons et on impute
# les trip_count manquants par 1.

print("\n" + "=" * 60)
print("ÉTAPE 2 — Nettoyage et standardisation")
print("=" * 60)

cols_keep = ['route_id', 'route_short_name', 'route_long_name',
             'route_type', 'trip_count', 'service_type', 'country']
df = df[cols_keep].copy()

# Construire un nom de route propre depuis long_name ou short_name
df['route_name'] = (
    df['route_long_name']
    .fillna(df['route_short_name'])
    .fillna('Unknown')
    .str.strip()
)

# Supprimer les routes sans nom identifiable
before = len(df)
df = df[df['route_name'] != 'Unknown'].copy()
print(f"✅ Routes sans nom supprimées : {before - len(df)}")

# Supprimer les doublons sur route_id + country + service_type
before = len(df)
df = df.drop_duplicates(subset=['route_id', 'country', 'service_type'])
print(f"✅ Doublons supprimés        : {before - len(df)}")

# Imputer trip_count manquant par 1
df['trip_count'] = df['trip_count'].fillna(1).astype(int)

print(f"\n✅ Shape après nettoyage : {df.shape}")

# ─────────────────────────────────────────────
# ÉTAPE 3 — FILTRAGE INTERCITY UNIQUEMENT
# ─────────────────────────────────────────────
# On filtre pour ne garder que les vrais trains intercity.
# En GTFS : route_type 2 = Rail, 100 = Railway, 109 = Suburban.
# On exclut la Suisse dont les données incluent bus/trams/métros
# locaux et représentaient 96% du dataset (146k sur 151k routes).

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

# ─────────────────────────────────────────────
# ÉTAPE 4 — CONSTRUCTION DE LA CIBLE is_underserved
# ─────────────────────────────────────────────
# Score de risque combinant trois signaux réels :
# - Fréquence faible (trip_count < médiane) : 50% du score
# - Train de nuit : 30% du score
# - Pays à faible part modale ferroviaire : 20% du score
# Un bruit aléatoire est ajouté pour éviter une prédiction
# parfaite et rendre le problème réaliste.
# Seuil 0.75 → balance 51/49

print("\n" + "=" * 60)
print("ÉTAPE 4 — Construction de la cible is_underserved")
print("=" * 60)

# Part modale ferroviaire nécessaire ici pour le calcul du score
rail_modal = {'FR': 11.1, 'DE': 9.9, 'DK': 7.9, 'EU': 8.4}
df['rail_modal_share'] = df['country'].map(rail_modal)

median_by_service = df.groupby('service_type')['trip_count'].transform('median')
rail_mean = df['rail_modal_share'].mean()

risk_score = (
    (df['trip_count'] < median_by_service).astype(int) * 0.5 +
    (df['service_type'] == 'night').astype(int) * 0.3 +
    (df['rail_modal_share'] < rail_mean).astype(int) * 0.2
)

noise = np.random.uniform(0, 0.4, len(df))
df['is_underserved'] = (risk_score + noise > 0.75).astype(int)

print("Distribution is_underserved :")
print(df['is_underserved'].value_counts())
print(df['is_underserved'].value_counts(normalize=True).mul(100).round(1))
print("\nTaux de sous-desserte par groupe :")
print(df.groupby(['country', 'service_type'])['is_underserved'].mean().mul(100).round(1))

# ─────────────────────────────────────────────
# ÉTAPE 5 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
# On crée des features supplémentaires à partir des données
# existantes et de sources externes réelles.

print("\n" + "=" * 60)
print("ÉTAPE 5 — Feature engineering")
print("=" * 60)

# Part modale ferroviaire par pays (Eurostat tran_hv_psmod, 2023)
# Source réelle — % de voyageurs qui choisissent le train par pays
# Hypothèse : pays avec faible part modale → plus de routes sous-desservies
# Feature indépendante de trip_count
rail_modal = {
    'FR': 11.1,
    'DE': 9.9,
    'DK': 7.9,
    'EU': 8.4
}
df['rail_modal_share'] = df['country'].map(rail_modal)

# Encodage service_type : day=0, night=1
df['type_encoded'] = df['service_type'].map({'day': 0, 'night': 1})

# Encodage pays
df['country_encoded'] = df['country'].map({'DE': 0, 'DK': 1, 'EU': 2, 'FR': 3})

# Route internationale (Eurostar, ÖBB, EC)
df['is_international'] = (df['country'] == 'EU').astype(int)

# Log du trip_count pour réduire l'asymétrie forte de la distribution
df['log_trip_count'] = np.log1p(df['trip_count'])

# Catégorie de fréquence pour l'analyse exploratoire
df['frequency_category'] = pd.cut(
    df['trip_count'],
    bins=[0, 2, 7, 30, float('inf')],
    labels=['très_faible', 'faible', 'moyen', 'élevé']
)

print(f"✅ Features créées")
print(f"Shape : {df.shape}")

# ─────────────────────────────────────────────
# ÉTAPE 6 — VALIDATION FINALE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÉTAPE 6 — Validation finale")
print("=" * 60)

print(f"Shape finale : {df.shape}")
missing = df.isnull().sum()
print(f"\nValeurs manquantes :")
print(missing[missing > 0] if len(missing[missing > 0]) else "✅ Aucun missing")
print(f"\nDoublons : {df.duplicated().sum()}")

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