# ⚖️ Analyse des biais du modèle

## 🎯 Objectif

Identifier les biais potentiels dans le modèle de prédiction des sous-dessertes.

---

## ⚠️ Sources de biais

### 🔹 Données simulées

- approximation du comportement réel
- hypothèses simplifiées

---

### 🔹 Déséquilibre des classes

- peu de lignes sous-desservies
- modèle biaisé vers la classe majoritaire

---

### 🔹 Règle métier

- définition simplifiée de "sous-desservie"
- dépendance à des seuils arbitraires

---

## 📉 Impact

- erreurs sur certains cas limites
- difficulté à généraliser

---

## ✅ Solutions mises en place

- rééquilibrage du dataset (oversampling)
- amélioration de la règle métier
- introduction de bruit

---

## 🚀 Améliorations possibles

- données réelles
- meilleure définition métier
- validation par experts terrain

---

## 📊 Conclusion

Le modèle présente des biais réalistes mais maîtrisés.

➡️ important de garder une approche critique  
➡️ modèle utile mais non parfait