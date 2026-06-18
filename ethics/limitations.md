# Limitations du modèle — ObRail MSPR 2025-2026

## 1. Données simulées

Plusieurs variables clés utilisées pour entraîner le modèle n'étaient pas disponibles dans les données réelles. Elles ont donc été simulées ou estimées à partir de règles métier :

- **`passengers_estimated`** : nombre de passagers estimé à partir de la capacité et de règles heuristiques. Cette variable ne reflète pas la fréquentation réelle des lignes.
- **`capacity`** : nombre de places estimé selon le type de train (jour ou nuit). Les capacités réelles varient selon le matériel roulant et l'opérateur.
- **`load_factor`** : taux de remplissage calculé à partir des deux variables ci-dessus. Sa fiabilité dépend directement de la qualité des estimations.

Ces données simulées introduisent un biais structurel dans le modèle. Les prédictions doivent être interprétées avec précaution et ne peuvent pas se substituer à une analyse basée sur des données réelles.

---

## 2. Déséquilibre des classes

Le jeu de données présente un déséquilibre important entre les lignes desservies et sous-desservies :

- **80.6%** des lignes sont classées comme non sous-desservies
- **19.4%** des lignes sont classées comme sous-desservies

Ce déséquilibre a pour conséquence que le modèle est plus performant pour détecter les lignes correctement desservies que les lignes sous-desservies. Le recall sur la classe minoritaire (sous-desservie) reste limité malgré les techniques de rééquilibrage appliquées (`scale_pos_weight`).

---

## 3. Data leakage (fuite de données)

Lors du développement, un phénomène de data leakage a été identifié : les variables `load_factor` et `passengers_estimated` étaient directement liées à la construction de la variable cible `is_underserved`. Leur présence dans les features permettait au modèle d'accéder indirectement à la réponse qu'il devait prédire, produisant artificiellement des scores parfaits (accuracy = 1.0).

Ce problème a été corrigé en retirant ces variables du feature set d'entraînement. Le modèle s'appuie désormais sur des variables réellement indépendantes.

---

## 4. Généralisation limitée

Le modèle a été entraîné sur des données principalement issues de pays d'Europe occidentale (France, Suisse, Allemagne, Autriche). La couverture géographique est inégale, ce qui peut limiter la capacité du modèle à généraliser sur des pays moins représentés dans les données (pays d'Europe de l'Est, pays baltes, etc.).

---

## 5. Absence de données temporelles

Le modèle ne prend pas en compte l'évolution des lignes dans le temps. Les données utilisées correspondent à une photographie statique du réseau ferroviaire européen. Des variations saisonnières, des événements ponctuels ou des évolutions du réseau ne sont pas reflétés dans les prédictions.

---

## 6. Performances du modèle final

Le modèle final (LightGBM optimisé) présente les performances suivantes sur le jeu de test :

| Métrique | Valeur |
|---|---|
| Accuracy | 0.75 |
| F1 (pondéré) | 0.78 |
| ROC-AUC | 0.88 |
| Recall (sous-desservie) | 0.97 |
| Precision (sous-desservie) | 0.44 |

La précision sur la classe sous-desservie (0.44) indique que le modèle génère des faux positifs — des lignes prédites comme sous-desservies alors qu'elles ne le sont pas. Ce comportement est acceptable dans un contexte exploratoire mais doit être amélioré avant tout déploiement en production.

---

## 7. Plan d'amélioration

- Intégrer des données réelles de fréquentation et de capacité auprès des opérateurs ferroviaires (SNCF, DB, ÖBB, etc.)
- Enrichir le dataset avec des données Eurostat sur la mobilité et la démographie
- Tester des architectures plus avancées (réseaux de neurones, gradient boosting avec features temporelles)
- Déployer le modèle sur une infrastructure cloud pour permettre un réentraînement régulier avec de nouvelles données