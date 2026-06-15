# 📒 Notebooks — ObRail (détection des sous-dessertes)

Pipeline d'analyse, du brut au modèle évalué. **Un notebook = un·e propriétaire** (pour éviter les conflits de merge).

| Notebook | Contenu | Propriétaire |
|---|---|---|
| `01_eda.ipynb` | Analyse exploratoire (qualité, cible, features vs cible) | Louis |
| `02_feature_engineering.ipynb` | Construction des features | Charlotte |
| `03_models.ipynb` | Modèles (RF / XGBoost) | Charlotte + Louis |
| `04_optimization.ipynb` | Optimisation des hyperparamètres | Charlotte |
| `05_explainability.ipynb` | SHAP / interprétabilité | Charlotte |
| `06_evaluation.ipynb` | Évaluation (métriques, courbes) | Louis |

## ▶️ Lancer

```bash
pip install -r ../requirements.txt
jupyter lab        # ou jupyter notebook
```

Les notebooks lisent les données depuis des **chemins relatifs** (`../data/...`) et sauvegardent les figures dans `../evaluation/plots/`. `random_state = 42` partout pour la reproductibilité.

---

## ⚙️ `nbstripout` — à installer OBLIGATOIREMENT (toute l'équipe)

> ⚠️ **Pourquoi** : un notebook `.ipynb` est un gros fichier JSON contenant le code **et** toutes les sorties (texte, images encodées). Quand deux personnes modifient le même notebook, git tente de fusionner ce JSON et le **corrompt** (déjà arrivé sur `01_eda.ipynb`). `nbstripout` retire automatiquement les **sorties** au moment du commit : les notebooks versionnés ne contiennent plus que **code + markdown**.

**Bénéfices** : diffs lisibles, **quasi plus de conflits/corruption**, repo léger.

### Installation (chaque membre, une seule fois, dans le repo)

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

> Le `.gitattributes` est déjà versionné (il déclare le filtre), mais **chaque membre doit quand même lancer les 2 commandes ci-dessus** : la *définition* du filtre vit dans `.git/config` (local, non partagé). Sans ça, le filtre ne s'applique pas chez toi.

### Vérifier que c'est actif

```bash
git config --get filter.nbstripout.clean   # doit renvoyer une commande nbstripout
```

### Conséquence à connaître
Les notebooks versionnés **n'affichent plus les sorties/figures** (il faut faire *Restart & Run All* pour les régénérer). Ce n'est pas un problème : **les figures livrables sont sauvegardées séparément** dans `../evaluation/plots/`.

---

## 🛟 Règles anti-conflit (important)

1. **Toujours `git pull` avant** d'ouvrir/éditer un notebook.
2. **Une branche dédiée** par tâche, puis Pull Request.
3. **Ne jamais éditer un notebook qui n'est pas le tien.**
4. **Ne jamais résoudre un conflit de `.ipynb` à la main** dans un éditeur de texte → c'est ce qui a corrompu `01_eda.ipynb`. En cas de conflit : garder une version, régénérer l'autre.
