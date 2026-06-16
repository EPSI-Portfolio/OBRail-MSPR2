"""
features.py — ObRail MSPR 2025-2026
Auteure : Charlotte

Rôle :
Construction du feature set final et des splits train/test à partir
de routes_processed.csv. Version script du notebook 02_feature_engineering.

Entrée  : data/processed/routes_processed.csv
Sorties : data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
          data/processed/scaler.joblib
          data/processed/variables_retenues.csv

Usage :
    python src/features.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 42
np.random.seed(SEED)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_underserved"

# Colonnes sources de la cible — exclues pour éviter la fuite de données
LEAKY_COLS = ["days_active", "distance_km"]

# Features numériques retenues
NUMERIC_FEATURES = [
    "rail_modal_share",
    "type_encoded",
    "is_international",
    "log_distance",
    "log_co2",
]

# Colonnes continues à normaliser (les binaires ne le sont pas)
COLS_TO_SCALE = ["rail_modal_share", "log_distance", "log_co2"]


def main() -> None:
    print("=" * 60)
    print("ÉTAPE 1 — Chargement")
    print("=" * 60)

    df = pd.read_csv(
        PROCESSED_DIR / "routes_processed.csv",
        dtype={"days_of_week": str},
    )
    print(f"✅ Chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Encodage one-hot du pays de départ")
    print("=" * 60)

    # Le pays est nominal — one-hot évite un ordre artificiel.
    # drop_first=True évite la multicolinéarité parfaite (régression logistique).
    country_dummies = pd.get_dummies(
        df["departure_country"],
        prefix="country",
        drop_first=True,
        dtype=int,
    )
    print(f"✅ Colonnes one-hot créées : {country_dummies.shape[1]}")

    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Construction du feature set final")
    print("=" * 60)

    X = pd.concat([df[NUMERIC_FEATURES], country_dummies], axis=1)
    y = df[TARGET]

    print(f"✅ Feature set : {X.shape[1]} colonnes, {X.shape[0]} lignes")
    print(f"   Valeurs manquantes : {X.isnull().sum().sum()}")

    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Tableau des variables retenues")
    print("=" * 60)

    variables_table = pd.DataFrame([
        {"Variable": "rail_modal_share", "Type": "Numérique continu",
         "Source": "Eurostat",
         "Justification": "Part modale ferroviaire du pays de départ"},
        {"Variable": "type_encoded", "Type": "Binaire",
         "Source": "is_night_train",
         "Justification": "Trains de nuit sous-desservis à 75.4% vs 17.9%"},
        {"Variable": "is_international", "Type": "Binaire",
         "Source": "departure vs arrival country",
         "Justification": "Routes internationales sous-desservies à 94.4%"},
        {"Variable": "log_distance", "Type": "Numérique continu",
         "Source": "log(1 + distance_km)",
         "Justification": "Distance opérationnelle (caveat : lié au seuil 100km)"},
        {"Variable": "log_co2", "Type": "Numérique continu",
         "Source": "log(1 + co2_per_pkm)",
         "Justification": "Empreinte carbone par passager-km"},
        {"Variable": "country_XX (×33)", "Type": "Binaire (one-hot)",
         "Source": "departure_country",
         "Justification": "Effet pays — disparités observées en EDA"},
    ])
    variables_table.to_csv(PROCESSED_DIR / "variables_retenues.csv", index=False)
    print("✅ Tableau sauvegardé → data/processed/variables_retenues.csv")

    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Split train/test stratifié 80/20")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    print(f"✅ X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    print(f"   Taux underserved train : {y_train.mean()*100:.1f}%")
    print(f"   Taux underserved test  : {y_test.mean()*100:.1f}%")

    print("\n" + "=" * 60)
    print("ÉTAPE 6 — Normalisation (fit sur train uniquement)")
    print("=" * 60)

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])
    print("✅ Normalisation appliquée sur :", COLS_TO_SCALE)

    print("\n" + "=" * 60)
    print("ÉTAPE 7 — Sauvegarde")
    print("=" * 60)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")

    print("✅ Splits + scaler sauvegardés dans data/processed/")
    print("\n🎉 features.py terminé avec succès !")


if __name__ == "__main__":
    main()