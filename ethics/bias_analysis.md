# Analyse des biais du modèle
**Projet** : ObRail MSPR 2025-2026  
**Date** : Juin 2026

---

## 1. Biais lié à la construction de la cible

La variable cible `is_underserved` a été construite à partir de la règle métier suivante :  
`days_active <= 3 AND distance_km > 100`

Cette règle introduit un biais structurel : le modèle apprend partiellement une formule plutôt que des patterns opérationnels réels. L'analyse SHAP confirme ce biais — `log_distance` domine les décisions du modèle avec une importance de 5.17, contre 0.19 pour la feature suivante (`rail_modal_share`). Le ratio est de 27:1, ce qui signifie que la distance est de loin le facteur le plus déterminant.

**Impact concret** : le modèle flaggue systématiquement les routes longues comme sous-desservies, même si elles circulent fréquemment. Cela explique le taux élevé de faux positifs (1 213 sur 5 040 routes de test).

**Recommandation** : remplacer la règle arbitraire par une cible construite à partir de données GTFS réelles sur la fréquence hebdomadaire des dessertes.

---

## 2. Biais lié au déséquilibre de classes

Le dataset présente un déséquilibre 4.16:1 :
- Non sous-desservies : 20 313 routes (80.6%)
- Sous-desservies : 4 887 routes (19.4%)

Sans correction, un modèle naïf prédisant toujours 0 obtiendrait 80.6% d'accuracy — ce qui est trompeur. Ce biais a été corrigé via `scale_pos_weight=4.16` dans LightGBM, qui pénalise davantage les erreurs sur la classe minoritaire. Le F1-score a été retenu comme métrique principale plutôt que l'accuracy pour cette raison.

**Impact résiduel** : malgré la correction, la précision sur la classe sous-desservie reste à 0.44 — le modèle génère encore beaucoup de faux positifs.

---

## 3. Biais géographique

Le dataset est déséquilibré géographiquement :
- France : 6 960 routes (27.6%)
- Espagne : 4 789 routes (19.0%)
- Allemagne : 4 429 routes (17.6%)

Ces trois pays représentent 64% des données. Les pays avec peu de routes (Monténégro, Macédoine, Albanie) sont très sous-représentés. Les prédictions pour ces pays sont moins fiables car le modèle a vu peu d'exemples les concernant lors de l'entraînement.

**Recommandation** : enrichir le dataset avec des sources GTFS complémentaires pour les pays sous-représentés.

---

## 4. Biais lié aux données simulées

Certaines features numériques (`co2_per_pkm`, `emissions_co2`) ont été générées par simulation. L'analyse de corrélation en EDA a montré des corrélations quasi nulles de `log_co2` avec la cible (-0.01). L'importance SHAP de `log_co2` est faible (0.14), ce qui confirme que cette feature n'apporte pas de signal réel au modèle.

**Impact** : la feature est présente dans le modèle sans contribuer significativement. Elle n'introduit pas de biais actif mais représente une opportunité d'amélioration.

**Recommandation** : remplacer les données simulées par des données réelles issues d'ADEME ou de l'Agence Européenne de l'Environnement.

---

## 5. Ce qui a été mis en place pour limiter les biais

- `scale_pos_weight=4.16` dans LightGBM pour corriger le déséquilibre de classes
- Suppression de `days_active` et `distance_km` du feature set pour éviter la fuite directe de données
- F1-score retenu comme métrique principale plutôt que l'accuracy
- Validation croisée stratifiée 5-fold pour garantir la représentativité des splits
- Analyse SHAP pour identifier et documenter la dominance de `log_distance`
- Analyse LIME sur les faux positifs pour comprendre les erreurs du modèle

---

## 6. Conclusion

Le modèle présente des biais identifiés, documentés et partiellement corrigés. Le biais principal — la dominance de `log_distance` liée à la règle de construction de la cible — est une limite fondamentale qui ne peut pas être résolue par le tuning du modèle seul. Elle nécessite une révision de la définition de la cible `is_underserved` à partir de données de fréquence réelles.

Le modèle reste utile dans son état actuel pour identifier des routes candidates à l'analyse, mais ses prédictions doivent être interprétées avec cette limite en tête et vérifiées par des experts terrain avant toute décision opérationnelle.