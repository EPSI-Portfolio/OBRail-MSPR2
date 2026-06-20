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
