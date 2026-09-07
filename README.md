# Adult Income Classifier — Pipelines + MLflow

Projet MLOps (M1) : pipeline de classification tabulaire reproductible avec
scikit-learn (`Pipeline` + `ColumnTransformer`), tuning par validation croisée,
et **suivi complet dans MLflow** (paramètres, métriques, artefacts, registre de modèles).
Le modèle est exposé en batch (`predict.py`) et via une API FastAPI conteneurisée.

## Le dataset : UCI Adult Income

Extrait du recensement américain de 1994 ([UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult)).
Tâche : prédire si une personne gagne **plus de 50 000 $ par an** (`>50K`) ou non.

| | |
|---|---|
| Lignes | 48 842 (adult.data + adult.test concaténés, split contrôlé par la config) |
| Features numériques | `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week` |
| Features catégorielles | `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` |
| Classe positive | ~24 % (`>50K`) → dataset déséquilibré, d'où ROC-AUC + courbe PR |

Particularités gérées par `src/data.py` et `src/utils.py` :
- fichiers sans en-tête, avec espaces avant chaque valeur ;
- valeurs manquantes encodées `?` (dans `workclass`, `occupation`, `native_country`) → `NaN`, imputées dans le pipeline ;
- la cible du fichier de test a un point final (`>50K.`) → normalisée ;
- `education` est ignorée car redondante avec `education_num` (déjà ordinale).

## Structure du repo

```
MLOps-project-M1/
├─ configs/
│  ├─ config.yaml            # logreg (défaut) : chemins, cible, split, features, grille, CV, MLflow
│  └─ config_rf.yaml         # variante random forest
├─ src/
│  ├─ data.py                # téléchargement UCI -> data/raw.csv
│  ├─ pipeline.py            # ColumnTransformer + modèle (logreg | random_forest | gradient_boosting)
│  ├─ train.py               # GridSearchCV + MLflow autolog + log_model + registre (alias `staging`)
│  ├─ evaluate.py            # métriques finales, ROC/PR/confusion, predictions.csv, métriques par sous-groupe
│  ├─ predict.py             # inférence batch CSV -> CSV
│  ├─ app.py                 # microservice FastAPI (/health, /predict)
│  └─ utils.py               # config, .env, MLflow setup, nettoyage, split, plots
├─ tests/                    # pytest : pipeline, utils, API
├─ examples/person.json      # payload d'exemple pour l'API
├─ data/                     # gitignoré
├─ artifacts/                # gitignoré : model.joblib, plots, predictions.csv, train_run.json
├─ Makefile · Dockerfile · requirements.txt · pyproject.toml · .env.example
└─ .github/workflows/ci.yml  # ruff + pytest
```

## Quickstart

Prérequis : Python 3.11 (`brew install python@3.11` sur macOS) et, pour la partie conteneur, Docker.

```bash
make init                 # venv Python 3.11 + dépendances (make init PYTHON=python3.12 pour changer)
cp .env.example .env      # MLFLOW_TRACKING_URI=sqlite:///mlflow.db, MLFLOW_EXPERIMENT_NAME=adult-income
make data                 # télécharge le dataset dans data/raw.csv
make train                # GridSearchCV (5-fold stratifié, roc_auc) + tracking + registre
make evaluate             # plots + predictions.csv + métriques par sous-groupe -> MLflow
make ui                   # MLflow UI sur http://127.0.0.1:5000
make test                 # pytest
make lint                 # ruff
```

Autres cibles utiles :

```bash
make train CONFIG=configs/config_rf.yaml               # entraîner la variante random forest
make evaluate                                          # évalue artifacts/model.joblib
.venv/bin/python -m src.evaluate --model "models:/AdultIncomeClassifier@staging"   # depuis le registre
make predict INPUT=data/new.csv OUTPUT=artifacts/scored.csv
make serve                                             # API sur http://localhost:8000/docs
make build && make docker-serve                        # même API dans Docker
```

## Ce que MLflow trace

Chaque `make train` crée un run parent (`<model_type>-<timestamp>`) contenant :

- **Params** : hyperparamètres du meilleur estimateur (autolog), tailles train/test, config CV.
- **Runs enfants** : un run par combinaison de la grille (`GridSearchCV` autolog) + `cv_results.csv`.
- **Métriques** : `cv_best_roc_auc`, et sur le jeu de test tenu à l'écart : `test_roc_auc`,
  `test_average_precision`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`.
- **Artefacts** : config YAML utilisée, `feature_importance.png`, modèle MLflow (signature + input example),
  plots d'entraînement générés par autolog.
- **Registre** : le modèle est enregistré sous `AdultIncomeClassifier`, avec l'alias `staging`
  (et le stage legacy `Staging` pour la compatibilité avec la consigne).

Chaque `make evaluate` crée un run `evaluate` lié au run d'entraînement (tag `train_run_id`) avec :
`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `classification_report.json`,
`predictions.csv` (pour l'analyse d'erreurs) et `subgroup_metrics.csv`.

## Résultats

Split stratifié 80/20, seed 42, CV 5-fold stratifiée sur `roc_auc`.

| Modèle (config) | Meilleurs params | CV ROC-AUC | Test ROC-AUC | Test AP | Test F1 |
|---|---|---|---|---|---|
| Logistic regression (`config.yaml`) | C=0.1, l2, lbfgs | 0.905 | 0.904 | 0.761 | 0.655 |
| Random forest (`config_rf.yaml`) | 200 arbres, max_depth=20, min_samples_leaf=1 | 0.918 | **0.921** | 0.814 | 0.694 |

Chaque `make train` enregistre une nouvelle version et lui donne l'alias `staging` (la dernière version entraînée est donc celle servie via `models:/AdultIncomeClassifier@staging`). Le random forest est le meilleur des deux et la version courante en `staging`.

### Note sur les sous-groupes

Le dataset contient des attributs sensibles (`sex`, `race`). `evaluate.py` calcule le ROC-AUC et le
taux de positifs par sous-groupe (`subgroup_metrics.csv`, aussi loggés comme métriques MLflow).
Le taux de `>50K` est très différent entre groupes (ex. ~11 % chez les femmes contre ~30 % chez les hommes),
ce qui reflète les données de 1994 : un tel modèle ne devrait pas être utilisé pour une décision réelle
sans une analyse d'équité approfondie.

## API

```bash
make serve
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @examples/person.json
# {"label":">50K","score":0.91}
```

Les champs catégoriels sont optionnels (imputés par le pipeline). La validation Pydantic renvoie 422
sur une valeur hors plage, et l'API renvoie 503 si aucun modèle n'est entraîné.

## Docker

```bash
make build                                   # image adult-income-classifier:latest (~1.4 GB, python:3.11-slim)
make docker-serve                            # sert artifacts/model.joblib sur :8000
make docker-train                            # entraîne dans le conteneur (monte data/, artifacts/, mlflow.db)
```

## Choix techniques

- **Tout passe par la config YAML** (features, grille, CV, MLflow) : changer de modèle ne touche pas au code.
- **Split avant tuning** : le jeu de test n'est jamais vu par `GridSearchCV`, les métriques `test_*` sont honnêtes.
- **`handle_unknown="ignore"`** sur le one-hot : l'API accepte des catégories inédites sans planter.
- **Backend SQLite** pour MLflow (`sqlite:///mlflow.db`) : nécessaire au registre de modèles, contrairement au store fichier.
- **Alias plutôt que stages** : les stages sont dépréciés depuis MLflow 2.9, on utilise `@staging` et on garde le stage en fallback.
