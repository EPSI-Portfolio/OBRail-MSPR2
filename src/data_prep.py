"""
data_prep.py — ObRail MSPR 2025-2026
Auteure : Charlotte

Rôle :
Construction du dataset de modélisation à partir
du dataset ferroviaire européen filtré.

Source :
data/raw/routes_europe_filtered.csv

Résultat :
data/processed/routes_processed.csv

Corrections v2 :
- Lecture de days_of_week forcée en str pour éviter la conversion float
  (ex: '1111111' lu comme 1111111.0 puis tronqué à 8 chars après nettoyage)
- Suppression de days_active et distance_km du jeu de features final
  pour éviter la fuite de données (is_underserved est dérivé de ces deux colonnes)
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ─── Reproductibilité ────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

# ─── Chemins ─────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "routes_europe_filtered.csv"
OUTPUT_FILE = PROCESSED_DIR / "routes_processed.csv"

# ─── Référentiel part modale ferroviaire par pays (source : Eurostat) ────────

RAIL_MODAL = {
    "FR": 11.1, "DE": 9.9,  "DK": 7.9,  "GB": 9.2,  "IT": 6.8,  "ES": 6.5,
    "PL": 8.1,  "NL": 11.5, "BE": 8.3,  "AT": 12.1, "SE": 9.8,  "CH": 17.9,
    "CZ": 9.1,  "HU": 9.3,  "SK": 8.7,  "PT": 5.2,  "GR": 2.1,  "RO": 5.8,
    "HR": 4.3,  "SI": 4.8,  "FI": 5.9,  "NO": 6.2,  "IE": 3.8,  "LU": 7.1,
    "LT": 4.2,  "EE": 4.5,  "LV": 5.1,  "BG": 4.9,  "RS": 3.2,  "ME": 2.8,
    "MK": 2.1,  "AL": 1.9,  "UA": 6.8,  "MD": 4.1,
}

BOOL_MAP = {
    "true": True,  "false": False,
    "1":    True,  "0":     False,
    True:   True,  False:   False,
}

REQUIRED_COLS = [
    "route_id",
    "departure_country",
    "arrival_country",
    "days_of_week",
    "is_night_train",
    "distance_km",
    "co2_per_pkm",
]

# Colonnes dérivées de la cible — exclues du feature set final
# pour éviter toute fuite de données vers is_underserved
LEAKY_COLS = ["days_active", "distance_km"]


# ─── Helpers ─────────────────────────────────────────────────────────────────
"""
    Convertit une série days_of_week en nombre de jours actifs (int).

    Problème connu : pandas lit les valeurs binaires comme des flottants
    (ex : '1111111' → 1111111.0) ce qui ajoute un '.0' parasite.
    On nettoie d'abord ce suffixe avant de compter les '1'.

    Retourne NaN pour toute valeur non conforme (longueur ≠ 7 après nettoyage).
    
"""
    
def parse_days_of_week(series: pd.Series) -> pd.Series:
    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^01]", "", regex=True)
        .str.zfill(7)   # ← repad to 7 chars with leading zeros
    )
    return cleaned.apply(lambda x: x.count("1") if len(x) == 7 else np.nan)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:

    # ── Étape 1 : Chargement ─────────────────────────────────────────────────
    print("=" * 60)
    print("ÉTAPE 1 — Chargement des données")
    print("=" * 60)

    # dtype str sur days_of_week pour empêcher pandas de l'interpréter
    # comme un float et d'ajouter le '.0' parasite
    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        dtype={"days_of_week": str},
    )

    print(f"✅ Chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"\nColonnes : {df.columns.tolist()}")

    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Colonnes manquantes dans le fichier source : {missing_required}")

    print("\nRépartition is_night_train :")
    print(df["is_night_train"].value_counts(dropna=False))
    print(f"\nPays de départ distincts : {df['departure_country'].nunique()}")

    # ── Étape 2 : Nettoyage ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Nettoyage")
    print("=" * 60)

    before = len(df)
    df = df.dropna(subset=[
        "departure_country", "arrival_country",
        "distance_km", "days_of_week", "is_night_train",
    ]).copy()
    print(f"✅ Lignes supprimées (valeurs clés manquantes) : {before - len(df):,}")

    df["is_night_train"] = (
        df["is_night_train"]
        .astype(str).str.strip().str.lower()
        .map(BOOL_MAP)
    )

    before = len(df)
    df = df.dropna(subset=["is_night_train"]).copy()
    print(f"✅ Lignes supprimées (is_night_train invalide)  : {before - len(df):,}")

    df["service_type"] = df["is_night_train"].map({True: "night", False: "day"})

    before = len(df)
    df = df.drop_duplicates(subset=[
        "route_id", "departure_country", "arrival_country",
        "days_of_week", "is_night_train",
    ]).copy()
    print(f"✅ Doublons supprimés                           : {before - len(df):,}")
    print(f"✅ Shape après nettoyage                        : {df.shape}")

    print("\nRépartition service_type :")
    print(df["service_type"].value_counts())

    # ── Étape 3 : Extraction des jours actifs ────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Extraction des jours actifs")
    print("=" * 60)

    print("\nAperçu days_of_week bruts (10 premières valeurs) :")
    print(df["days_of_week"].head(10).tolist())

    df["days_active"] = parse_days_of_week(df["days_of_week"])

    invalid_mask = df["days_active"].isna()
    print(f"\n✅ Valeurs invalides détectées : {invalid_mask.sum():,}")

    if invalid_mask.any():
        print("Exemples :")
        print(df.loc[invalid_mask, "days_of_week"].value_counts().head(10))

    before = len(df)
    df = df.dropna(subset=["days_active"]).copy()
    print(f"✅ Lignes supprimées (days_of_week invalide)    : {before - len(df):,}")

    df["days_active"] = df["days_active"].astype(int)

    before = len(df)
    df = df[df["days_active"] > 0].copy()
    print(f"✅ Routes sans circulation supprimées           : {before - len(df):,}")
    print(f"Shape après filtrage : {df.shape}")

    print("\nDistribution days_active :")
    print(df["days_active"].value_counts().sort_index())

    # ── Étape 4 : Construction de la cible ───────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Construction de la cible is_underserved")
    print("=" * 60)

    # Règle métier ObRail :
    #   route sous-desservie ⟺ ≤ 3 jours/semaine ET distance > 100 km
    # ATTENTION : days_active et distance_km sont des colonnes sources de la cible.
    # Elles sont conservées ici pour traçabilité mais EXCLUES du feature set (cf. étape 5).
    df["is_underserved"] = (
        (df["days_active"] <= 3) & (df["distance_km"] > 100)
    ).astype("int8")

    print("Distribution is_underserved :")
    print(df["is_underserved"].value_counts())
    print(f"\nTaux global de sous-desserte : {df['is_underserved'].mean() * 100:.1f} %")

    print("\nTaux par service_type :")
    print(
        df.groupby("service_type")["is_underserved"]
        .mean().mul(100).round(1)
        .rename("% underserved")
    )

    # ── Étape 5 : Feature engineering ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Feature engineering")
    print("=" * 60)

    # Part modale ferroviaire du pays de départ (proxy d'investissement réseau)
    df["rail_modal_share"] = (
        df["departure_country"].map(RAIL_MODAL).fillna(5.0)
    )

    # Encodage binaire du type de service
    df["type_encoded"] = df["service_type"].map({"day": 0, "night": 1}).astype("int8")

    # Route internationale ?
    df["is_international"] = (
        df["departure_country"] != df["arrival_country"]
    ).astype("int8")

    # Transformation log de la distance (réduit l'asymétrie)
    # NOTE : log_distance est conservé car il résume la distance de façon
    # moins directe que distance_km brute — mais reste potentiellement corrélé
    # à is_underserved. À surveiller lors de l'analyse de feature importance.
    df["log_distance"] = np.log1p(df["distance_km"])

    # CO₂ : imputation médiane puis transformation log
    df["co2_per_pkm"] = df["co2_per_pkm"].fillna(df["co2_per_pkm"].median())
    df["log_co2"] = np.log1p(df["co2_per_pkm"])

    # Encodage ordinal du pays de départ (top 10 + bucket "autres")
    top_countries = (
        df["departure_country"].value_counts().head(10).index.tolist()
    )
    country_map = {c: i for i, c in enumerate(top_countries)}
    df["country_encoded"] = (
        df["departure_country"]
        .map(country_map)
        .fillna(len(top_countries))
        .astype("int16")
    )

    print(f"✅ Features créées. Shape intermédiaire : {df.shape}")

    # ── Marquage des colonnes sources de la cible (fuite de données) ─────────
    print(f"\n⚠️  Colonnes exclues du feature set (dérivées de is_underserved) :")
    print(f"   {LEAKY_COLS}")
    print("   → Conservées dans le CSV pour traçabilité, à exclure lors du split train/test.")

    # ── Étape 6 : Validation finale ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 6 — Validation finale")
    print("=" * 60)

    print(f"Shape finale : {df.shape}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nValeurs manquantes :")
    print(missing if len(missing) > 0 else "✅ Aucune valeur manquante")

    print(f"\nDoublons résiduels : {df.duplicated().sum()}")
    print(f"\nColonnes finales :\n{df.columns.tolist()}")

    # Résumé feature set recommandé pour la modélisation
    FEATURE_COLS = [
        "rail_modal_share",
        "type_encoded",
        "is_international",
        "log_distance",   # voir note ci-dessus
        "log_co2",
        "country_encoded",
    ]
    print(f"\n📋 Feature set recommandé pour 02_feature_engineering.ipynb :")
    print(f"   {FEATURE_COLS}")
    print(f"   Cible : is_underserved")

    # ── Étape 7 : Sauvegarde ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 7 — Sauvegarde")
    print("=" * 60)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Sauvegardé → {OUTPUT_FILE}")
    print(f"   {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print("\n🎉 data_prep.py terminé avec succès !")


if __name__ == "__main__":
    main()