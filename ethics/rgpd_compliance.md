<<<<<<< HEAD
# Conformité RGPD
**Projet** : ObRail MSPR 2025-2026

---

## 1. Objectif

Analyser la conformité du projet ObRail au Règlement Général sur la Protection des Données (RGPD), conformément aux exigences du cahier des charges.

---

## 2. Données utilisées

Le projet repose exclusivement sur des données techniques relatives aux liaisons ferroviaires :
- caractéristiques des routes (distance, pays de départ et d'arrivée, type de service)
- jours de circulation hebdomadaire
- facteurs d'émission de CO2

Certaines variables (émissions, fréquentation) ont été partiellement simulées en l'absence de données réelles. Aucune donnée à caractère personnel n'est collectée ni traitée à aucune étape du pipeline.

---

## 3. Respect des principes RGPD

Le RGPD s'applique au traitement de données personnelles. Le projet ne traitant que des données techniques sur des infrastructures, il sort du périmètre direct du RGPD. Néanmoins, les principes fondamentaux ont été appliqués par bonne pratique :

- **Minimisation** : seules les variables nécessaires à la prédiction sont conservées dans le feature set. Les colonnes non pertinentes ou redondantes sont écartées.
- **Transparence** : les sources de données sont documentées et tracées (GTFS, Eurostat).
- **Traçabilité** : chaque transformation appliquée aux données est documentée dans `src/data_prep.py` et les notebooks.
- **Reproductibilité** : les graines aléatoires sont fixées (seed=42) et le pipeline est entièrement reproductible.

---

## 4. Sécurité des données

- Les fichiers de données brutes et les modèles entraînés ne sont pas versionnés sur GitHub (`.gitignore`) — ils sont générés     localement par l'exécution du pipeline.
- Le modèle est sauvegardé au format `joblib`, un format standard sans dépendance propriétaire.
- L'API n'expose aucune donnée stockée — elle ne fait que retourner une prédiction à partir des entrées fournies.

---

## 5. Risques potentiels en cas d'évolution

Le projet deviendrait soumis au RGPD si des données personnelles étaient introduites dans une future version, par exemple :
- des données de fréquentation réelle rattachées à des voyageurs identifiables
- l'intégration de données clients d'opérateurs ferroviaires

---

## 6. Recommandations pour une future version avec données réelles

- Recueillir le consentement des personnes concernées si des données voyageurs sont utilisées
- Anonymiser ou pseudonymiser toute donnée rattachable à un individu
- Sécuriser le stockage et restreindre les accès
- Documenter les traitements dans un registre conforme RGPD
- Réaliser une analyse d'impact (AIPD) si le traitement présente un risque élevé

---

## 7. Conclusion

Dans son état actuel, le projet ObRail est conforme au RGPD car il ne traite aucune donnée à caractère personnel — uniquement des données techniques relatives aux infrastructures ferroviaires. Les principes de minimisation, transparence et traçabilité ont néanmoins été appliqués par bonne pratique, ce qui facilitera la mise en conformité si des données personnelles venaient à être intégrées à l'avenir.
=======
# Conformité RGPD — ObRail MSPR 2025-2026

## 1. Contexte

Le projet ObRail vise à détecter les zones de sous-desserte ferroviaire en Europe à partir de données relatives aux trajets, distances, émissions CO2 et opérateurs ferroviaires. Ce document décrit les mesures prises pour garantir la conformité au Règlement Général sur la Protection des Données (RGPD - Règlement UE 2016/679).

---

## 2. Nature des données traitées

**Aucune donnée personnelle n'est collectée, traitée ou stockée dans ce projet.**

Les données utilisées sont exclusivement des données techniques et opérationnelles :

| Type de donnée | Exemple | Données personnelles ? |
|---|---|---|
| Trajets ferroviaires | Paris → Berlin | Non |
| Distance | 1 050 km | Non |
| Type de train | Jour / Nuit | Non |
| Émissions CO2 | 28.3 gCO2/pkm | Non |
| Pays de départ/arrivée | FR, DE | Non |
| Opérateur | SNCF, DB, ÖBB | Non |

Le projet ne traite aucune information permettant d'identifier directement ou indirectement une personne physique (nom, prénom, email, localisation individuelle, identifiant).

---

## 3. Principes RGPD appliqués

### 3.1 Minimisation des données
Seules les données strictement nécessaires à la détection des zones de sous-desserte ont été collectées et utilisées. Aucune donnée superflue n'est présente dans le dataset final.

### 3.2 Transparence
Les sources de données sont documentées et traçables :
- **OpenFlights** : données publiques sur les routes aériennes et aéroports
- **Eurostat** : statistiques officielles de l'Union Européenne
- **GTFS** : standard international de données ferroviaires publié par les opérateurs
- **Pipeline ETL** : données harmonisées issues du projet précédent ObRail MSPR1

### 3.3 Limitation de la finalité
Les données collectées sont utilisées uniquement dans le cadre de ce projet académique. Elles ne sont pas partagées avec des tiers ni utilisées à d'autres fins que l'analyse ferroviaire et l'entraînement du modèle IA.

### 3.4 Exactitude
Les données manquantes ont été imputées ou simulées selon des règles métier documentées. Les données simulées sont clairement identifiées dans la documentation technique (`limitations.md`).

### 3.5 Sécurité
- Les données sont stockées localement ou sur un dépôt GitHub privé accessible uniquement aux membres de l'équipe projet
- Aucune donnée sensible n'est exposée via l'API
- Les fichiers `.joblib` et `.csv` sont exclus du dépôt public via `.gitignore`

---

## 4. Données simulées et responsabilité

Certaines variables ont été simulées (fréquentation, capacité, taux de remplissage) en raison de leur indisponibilité ou de leur caractère confidentiel auprès des opérateurs. Ces simulations sont basées sur des hypothèses métier documentées et ne constituent pas une collecte de données réelles.

---

## 5. Intégration future de données réelles

Si le projet venait à intégrer des données réelles de fréquentation ou de capacité en production, les mesures suivantes devront être mises en place :

- **Anonymisation** des données si elles contiennent des informations individuelles (ex : données de billetterie)
- **Convention de traitement** avec les opérateurs ferroviaires fournisseurs de données
- **Registre des traitements** conformément à l'article 30 du RGPD
- **Analyse d'impact (AIPD)** si les données traitées présentent un risque élevé pour les droits et libertés des personnes
- **Durée de conservation** définie et documentée pour chaque type de donnée

---

## 6. Références réglementaires

- Règlement (UE) 2016/679 — RGPD : https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32016R0679
- CNIL — Intelligence artificielle : cadre éthique et bonnes pratiques : https://www.cnil.fr/fr/intelligence-artificielle
- Ethics Guidelines for Trustworthy AI (Commission européenne) : https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai

---

## 7. Conclusion

Le projet ObRail MSPR 2025-2026 respecte les principes fondamentaux du RGPD. En l'absence de données personnelles, le risque de violation de la vie privée est nul. Les bonnes pratiques de documentation, de traçabilité et de sécurité ont été appliquées tout au long du projet afin de garantir une démarche responsable et conforme aux réglementations européennes en vigueur.
>>>>>>> jeannette/data-prep
