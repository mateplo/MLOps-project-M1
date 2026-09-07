# Adult Income Classifier — Pipelines, MLflow & un cycle MLOps complet

Projet MLOps (M1) : pipeline de classification tabulaire reproductible avec scikit-learn
(`Pipeline` + `ColumnTransformer`), tuning par validation croisée, **suivi complet dans MLflow**
(paramètres, métriques, artefacts, lignée des données, registre de modèles avec promotion
champion/challenger), API FastAPI conteneurisée avec traçabilité du modèle servi, journal des
prédictions, détection de dérive (PSI) et stack de monitoring locale (Prometheus + Grafana).

```
 make data ──▶ make train ──▶ make promote ──▶ make export ──▶ make serve / docker compose
   UCI CSV      validate        champion ?        artifacts/       /predict  /health  /metrics
   + hash       GridSearchCV    (registry)        model.joblib          │
                seuil F1        alias             model_meta.json       ▼
                MLflow run      champion/                          logs/predictions.jsonl
                                challenger                                │
                                                                  make drift (PSI → MLflow)
```

## Le dataset : UCI Adult Income

Extrait du recensement américain de 1994 ([UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult)).
Tâche : prédire si une personne gagne **plus de 50 000 $ par an** (`>50K`) ou non.

| | |
|---|---|
| Lignes | 48 842 (adult.data + adult.test concaténés, split contrôlé par la config) |
| Features numériques | `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week` |
| Features catégorielles | `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country` |
| Classe positive | ~24 % (`>50K`) → déséquilibre, d'où ROC-AUC + courbe PR + seuil optimisé |

Particularités gérées par `src/data.py`, `src/validate.py` et `src/utils.py` :
- fichiers sans en-tête, espaces avant chaque valeur, `?` pour les valeurs manquantes → `NaN` imputé dans le pipeline ;
- cible du fichier de test avec un point final (`>50K.`) → normalisée ;
- **`fnlwgt` est retirée** : c'est un poids d'échantillonnage du recensement, pas une caractéristique de la personne ;
- **`education` est retirée** : redondante avec `education_num` (déjà ordinale) ;
- **`capital_gain` / `capital_loss`** passent par un `log1p` (distributions très asymétriques) ;
- **`native_country`** (41 modalités) : les catégories < 1 % sont regroupées par `OneHotEncoder(min_frequency=0.01)` ;
- les colonnes numériques sont castées en `float64` pour que la signature MLflow tolère un `NaN` à l'inférence.

## Structure du repo

```
MLOps-project-M1/
├─ configs/
│  ├─ config.yaml            # logreg (défaut) : données, règles de validation, features, grille, CV, MLflow, promotion
│  ├─ config_hgb.yaml        # HistGradientBoosting (meilleur modèle)
│  ├─ config_rf.yaml         # RandomForest + CalibratedClassifierCV (isotonic)
│  └─ config_ci.yaml         # run de fumée pour la CI (5k lignes, 2 folds)
├─ src/
│  ├─ data.py                # téléchargement UCI -> data/raw.csv
│  ├─ validate.py            # validation de schéma / plages / taux de positifs avant entraînement
│  ├─ pipeline.py            # ColumnTransformer (num | log1p | cat) + modèle (+ calibration optionnelle)
│  ├─ metrics.py             # métriques, seuil de décision optimal (F1)
│  ├─ train.py               # GridSearchCV + seuil OOF + MLflow (autolog, log_input, log_model) + registre `challenger`
│  ├─ promote.py             # champion/challenger : promotion seulement si meilleur sur le split de test
│  ├─ evaluate.py            # ROC/PR/confusion/calibration, predictions.csv, métriques par sous-groupe
│  ├─ export.py              # registre -> artifacts/ (model.joblib + model_meta.json) pour l'API
│  ├─ predict.py             # inférence batch CSV -> CSV
│  ├─ app.py                 # FastAPI : /predict, /health (identité du modèle), /metrics (Prometheus)
│  ├─ drift.py               # PSI entre données d'entraînement et requêtes servies -> MLflow
│  ├─ simulate_traffic.py    # envoie des lignes réelles (ou biaisées) à l'API
│  ├─ export_onnx.py         # joblib -> model.onnx + preprocess.json, validé contre sklearn
│  ├─ publish.py             # champion exporté (joblib + onnx) -> dépôt de modèle HF (model card, tag vN)
│  ├─ fetch_model.py         # dépôt HF -> artifacts/ (entrypoint de l'image serve)
│  ├─ deploy_space.py        # génère et pousse le Space HF statique (index.html + app.js + config.json)
│  └─ utils.py               # logging, config, MLflow, nettoyage, split, plots, model_meta.json
├─ tests/                    # unitaires (pipeline, utils, validation, drift, API) + bout en bout (train→promote→evaluate→export→drift)
├─ monitoring/               # prometheus.yml, provisioning Grafana + dashboard
├─ deploy/                   # space/ (site statique : README, index.html, app.js) + space.env (MODEL_REVISION)
├─ scripts/                  # release_check.sh, entrypoint.sh
├─ docker-compose.yml        # mlflow server + api (+ profils train / monitoring)
├─ Dockerfile                # multi-stage : cibles `serve` (≈540 Mo) et `train`
├─ Makefile · requirements*.txt · requirements*.lock · pyproject.toml · .env.example
└─ .github/workflows/ci.yml  # lint + tests, entraînement de fumée, build Docker (+ push GHCR sur tag)
```

## Quickstart

Prérequis : Python 3.11 (`brew install python@3.11` sur macOS) et Docker pour la partie conteneurs.

```bash
make init                 # venv Python 3.11 + dépendances (make init PYTHON=python3.12 pour changer)
cp .env.example .env      # MLFLOW_TRACKING_URI=sqlite:///mlflow.db, MLFLOW_EXPERIMENT_NAME=adult-income
make data                 # télécharge le dataset dans data/raw.csv
make validate             # (optionnel) vérifie le CSV contre les règles de la config
make all                  # train -> promote -> evaluate
make ui                   # MLflow UI sur http://127.0.0.1:5001 (5000 est pris par AirPlay sur macOS)
make test                 # pytest (28 tests, dont le bout en bout)
make lint                 # ruff
```

Entraîner d'autres modèles, puis laisser la promotion décider :

```bash
make train CONFIG=configs/config_hgb.yaml && make promote CONFIG=configs/config_hgb.yaml
make train CONFIG=configs/config_rf.yaml  && make promote CONFIG=configs/config_rf.yaml
make export               # tire le champion du registre vers artifacts/
make evaluate MODEL="models:/AdultIncomeClassifier@champion"
make predict INPUT=data/new.csv OUTPUT=artifacts/scored.csv
```

## Ce que MLflow trace

**Run d'entraînement** (`<model_type>-<timestamp>`) :
- **Lignée des données** : `mlflow.log_input` (dataset pandas avec digest) + params `data_sha256`, `data_bytes`, `data_rows`, `data_positive_rate`. Deux runs entraînés sur des CSV différents sont distinguables.
- **Params** : meilleurs hyperparamètres (autolog), features utilisées, `log1p`, `min_frequency`, config CV, **`decision_threshold`**.
- **Runs enfants** : un par combinaison de la grille + `cv_results.csv`.
- **Métriques** : `cv_best_roc_auc`, `cv_f1_at_threshold`, et sur le test tenu à l'écart : `test_roc_auc`, `test_average_precision`, `test_brier`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1` (au seuil optimisé) et `test_f1_at_0.5` pour comparaison.
- **Artefacts** : config YAML, `feature_importance.png`, modèle MLflow avec signature, exemple d'entrée et `metadata` (seuil, type, hash des données).
- **Registre** : nouvelle version de `AdultIncomeClassifier` avec l'alias **`challenger`** et des tags (`decision_threshold`, `test_roc_auc`, `data_sha256`, `model_type`).

**Run `promote-vN`** : la décision (promu ou non, scores comparés, raison) est tracée comme un run, et la version reçoit les tags `promoted_at` / `promotion_reason` ou `promotion_rejected_at`.

**Run `evaluate`** (lié au run d'entraînement par `train_run_id` et `model_version`) : `roc_curve.png`, `pr_curve.png` (avec le point du seuil), `confusion_matrix.png`, `calibration_curve.png`, `classification_report.json`, `predictions.csv`, `subgroup_metrics.csv`, métriques `roc_auc_<feature>_<groupe>` et `roc_auc_gap_<feature>`.

**Run `drift`** : `psi_<feature>` pour chaque feature, `psi_max`, `drift_report.csv`.

### Seuil de décision

Le seuil n'est plus 0,5. Après le GridSearch, les probabilités **out-of-fold** (`cross_val_predict`) servent à choisir le seuil qui maximise le F1 sur le train. Il est loggé dans MLflow, stocké dans les métadonnées du modèle et dans `artifacts/model_meta.json`, et **consommé tel quel par `evaluate.py`, `predict.py` et l'API**. Le gain sur le meilleur modèle : F1 de 0,719 à 0,728, et sur la régression logistique de 0,638 à 0,678.

### Promotion champion / challenger

Chaque `make train` enregistre une version `challenger`. `make promote` compare son `test_roc_auc` (même split de test, même seed) à celui du `champion` courant et ne déplace l'alias `champion` que si c'est strictement mieux (`mlflow.promotion.min_improvement` dans la config, `--force` pour outrepasser). L'API ne sert jamais un modèle qui n'a pas gagné cette comparaison, puisque `make export` tire le champion.

## Résultats

Split stratifié 80/20, seed 42, CV 5-fold stratifiée sur `roc_auc`, seuil optimisé sur le F1 out-of-fold.

| Modèle (config) | Meilleurs params | CV ROC-AUC | Test ROC-AUC | Test AP | Brier | Seuil | Test F1 | F1 à 0,5 | Promotion |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression (`config.yaml`) | C=1.0 | 0.899 | 0.896 | 0.729 | 0.108 | 0.33 | 0.678 | 0.638 | champion → battu |
| **HistGradientBoosting** (`config_hgb.yaml`) | lr=0.1, 400 iter, 15 feuilles, l2=1.0 | 0.928 | **0.930** | **0.833** | **0.087** | 0.38 | **0.728** | 0.719 | **champion** (v9) |
| RandomForest + calibration isotonique (`config_rf.yaml`) | 200 arbres, min_samples_leaf=5 | 0.918 | 0.920 | 0.810 | 0.093 | 0.36 | 0.713 | 0.695 | refusé (0.920 < 0.930) |

La régression logistique est passée de 0.904 à 0.896 de ROC-AUC en retirant `fnlwgt` et en regroupant les pays rares : c'est le prix d'un modèle plus honnête (pas de poids de sondage) et plus robuste aux catégories inédites. Le HGB compense largement.

### Note sur les sous-groupes

Le dataset contient des attributs sensibles (`sex`, `race`). `evaluate.py` calcule le ROC-AUC, le taux réel et le taux prédit de `>50K` par sous-groupe (`subgroup_metrics.csv`, métriques `roc_auc_gap_sex` / `roc_auc_gap_race` dans MLflow). Le taux de `>50K` est très différent entre groupes (~11 % chez les femmes contre ~30 % chez les hommes) et reflète les données de 1994 : un tel modèle ne devrait pas être utilisé pour une décision réelle sans une analyse d'équité approfondie.

## API

```bash
make serve
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @examples/person.json
# {"label":">50K","score":0.9749,"threshold":0.3809,"model_name":"AdultIncomeClassifier","model_version":"9","run_id":"8eec6466..."}
curl localhost:8000/health    # version, alias, run_id, seuil, hash des données d'entraînement, date
curl localhost:8000/metrics   # Prometheus : predictions_total{label,model_version}, prediction_latency_seconds, prediction_score
```

- Chaque réponse porte **la version du registre et le `run_id`** du modèle qui l'a produite.
- Chaque appel est ajouté à `logs/predictions.jsonl` (features, score, label, version, latence).
- Les champs catégoriels sont optionnels (imputés). Validation Pydantic → 422 ; pas de modèle → 503.

## Monitoring et dérive (100 % local)

```bash
make serve                        # dans un terminal
make simulate N=400               # 400 lignes réelles -> logs/predictions.jsonl
make drift                        # PSI par feature -> artifacts/drift_report.csv + run MLflow
make simulate-drift N=400         # échantillon biaisé (plus âgés, plus diplômés, gains x2, +10h/semaine)
make drift                        # -> hours_per_week passe en alerte (PSI 0.41)
```

Le PSI (Population Stability Index) compare la distribution de chaque feature entre les données
d'entraînement et les requêtes reçues : < 0,1 stable, 0,1–0,2 à surveiller, > 0,2 alerte. La colonne
`__predicted_positive_rate__` suit la dérive des prédictions elles-mêmes. Les valeurs sont loggées dans
MLflow, qui sert de tableau de bord historique sans infra supplémentaire.

Pour un vrai tableau de bord temps réel, le profil `monitoring` de docker-compose lance Prometheus
et Grafana avec un dashboard provisionné (débit, part de `>50K`, latence p95, version servie,
distribution des scores) :

```bash
make compose-monitoring           # mlflow :5001, api :8000, prometheus :9090, grafana :3000 (admin/admin)
make simulate N=1000              # puis regarder http://localhost:3000/d/adult-income-api
```

## Docker et docker-compose

`Dockerfile` multi-stage, dépendances installées depuis les **lock files** (`make lock` les régénère) :

| Cible | Contenu | Taille |
|---|---|---|
| `serve` (défaut) | venv minimal (`requirements-serve.lock` : scikit-learn, pandas, FastAPI, prometheus-client), non-root, healthcheck | ≈ 540 Mo |
| `train` | venv complet (`requirements.lock` : + MLflow, matplotlib) | ≈ 1,1 Go |

```bash
make build && make docker-serve   # API seule, sert artifacts/model.joblib
make docker-train                 # entraîne dans le conteneur ; le projet est monté à son chemin hôte (URIs MLflow absolues)
```

`docker-compose.yml` remplace le backend SQLite local par un **vrai serveur MLflow** (`--serve-artifacts`,
volume `mlflow-data`) et fait tourner l'API à côté :

```bash
make compose-up                   # mlflow server :5001 + api :8000
make compose-train                # data -> train -> promote -> evaluate -> export, contre le serveur (MLFLOW_TRACKING_URI=http://mlflow:5000)
make compose-monitoring           # + prometheus + grafana
make compose-down
```

Pour pointer le venv local vers ce serveur : `export MLFLOW_TRACKING_URI=http://localhost:5001`.

## Release

L'image n'est publiée sur GHCR que sur un tag `v*`. Avant de tagger, `make release-check` refuse la release si :
arbre Git non propre, CI non verte sur le commit courant, modèle exporté qui n'est pas le `champion`,
`test_roc_auc` sous `MIN_AUC` (0.92 par défaut), image qui ne sert pas exactement cette version
(`/health` doit renvoyer la même version et le même `run_id`), ou erreurs sur 200 requêtes réelles.

```bash
make release-check                     # toutes les vérifications, conteneur de test sur :8001
git tag -a v1.0.0 -m "AdultIncomeClassifier v9 (test_roc_auc 0.930)" && git push origin v1.0.0
docker pull ghcr.io/mateplo/mlops-project-m1:1.0.0     # une fois le package rendu public
```

## Déploiement Hugging Face (piloté par GitHub)

GitHub reste la source de vérité ; Hugging Face ne fait qu'héberger ce que la CI a produit.
Les Spaces Docker sont devenus payants (PRO), donc le Space est **statique** : le modèle tourne
**dans le navigateur** avec ONNX Runtime Web, exactement le pattern des labs transformers.js du cours.

```
GitHub (code, CI, gate, tag) ──▶ GHCR : image serve vX.Y.Z (API FastAPI, multi-arch)
        │
        └─▶ HF Hub, dépôt modèle ─────────▶ HF Space statique (index.html + app.js)
            model.joblib + model_meta.json     télécharge model.onnx + preprocess.json
            model.onnx + preprocess.json       et prédit côté client, seuil inclus
            tag vN = version du registre
```

- **Export ONNX** (`make export-onnx`, `src/export_onnx.py`) : trois briques scikit-learn n'ont pas de
  convertisseur, elles sont traitées à l'export : `FunctionTransformer(log1p)` par un convertisseur
  custom (`Log(Add(x,1))`), l'imputation des chaînes en traitant `"?"` comme manquant, et
  `OneHotEncoder(min_frequency)` en remplaçant l'encodeur par un one-hot classique dont les catégories
  sont les fréquentes plus le bucket `infrequent_sklearn`, le regroupement étant fait par le JavaScript
  d'après `preprocess.json`. L'export est **validé contre scikit-learn** sur 5 000 lignes
  (écart max 2e-7) ; `make release-check` refuse la release si l'écart dépasse 1e-4.
- **Dépôt de modèle HF** (`hf.model_repo`) : `make publish-model` y pousse le champion exporté
  (joblib pour l'API, ONNX + spec pour le navigateur, métadonnées, model card) avec un tag
  `v<version du registre>`. Refuse tout modèle qui n'est pas `champion`. Appelé en fin de `make release-check`.
- **Space HF statique** (`hf.space_repo`) : entièrement généré depuis `deploy/space/` par le job
  `deploy-space` de la CI sur chaque tag `v*`, avec un `config.json` qui pointe sur le dépôt de modèle
  et la révision de `deploy/space.env` (`main` = dernier champion publié, ou `v9` pour épingler).
  La page affiche la version du registre, le `run_id` et le seuil qu'elle utilise.
- **Image serve** : inchangée, c'est le déploiement « serveur » (API, journal, `/metrics`) pour un
  VPS ou un cloud. Si `MODEL_REPO` est défini, elle télécharge le modèle depuis le Hub au démarrage.

Mise en place, une seule fois : token HF en écriture dans le secret `HF_TOKEN` de l'environment GitHub
`huggingface`, et `hf.model_repo` / `hf.space_repo` au nom de ton compte (ou variables `HF_MODEL_REPO` /
`HF_SPACE_REPO`). Ensuite chaque release : `make release-check` → `git tag vX.Y.Z` → image GHCR,
Space redéployé, GitHub Release.

## Workflow Git

- **`main` est protégée** : pas de push direct, les checks `Lint + tests` et `Docker build (+ push on tag)` doivent passer avant fusion. Tout commit sur `main` est donc déployable.
- **Une branche courte par changement** (`feat/…`, `fix/…`, `docs/…`), fusionnée par pull request. Le template de PR rappelle la checklist et, quand le modèle change, demande le résultat de `make promote`.
- **Les tags `v*` marquent les releases** : `make release-check` puis `git tag -a vX.Y.Z` (voir ci-dessus). Pas de branche `develop` ni `release/*` : le tag et le gate jouent ce rôle.

```bash
git switch -c feat/ma-modif
# ... commits ...
git push -u origin feat/ma-modif        # puis ouvrir la PR sur GitHub, fusionner quand la CI est verte
```

## CI (GitHub Actions)

- **test** : `ruff check`, `ruff format --check`, `pytest` (avec le test bout en bout sur données synthétiques et registre SQLite temporaire).
- **smoke-train** : téléchargement du dataset réel, validation, `train → promote → evaluate → export` avec `config_ci.yaml`, artefacts uploadés.
- **release** (tag `v*`) : crée la GitHub Release avec les notes générées : commande `docker pull`, plateformes du manifeste, digest, commit, statut du déploiement HF, liens Space et modèle, et le changelog automatique des commits depuis le tag précédent.
- **docker** : build des cibles `serve` et `train` ; sur un tag `v*`, build **multi-plateforme** (`linux/amd64` + `linux/arm64`, donc Apple Silicon natif et HF Spaces) et push de l'image `serve` sur `ghcr.io/<owner>/mlops-project-m1`. `make build-multi` fait la même vérification en local.

## Choix techniques

- **Tout passe par la config YAML** : features, transformations, grille, CV, règles de validation, politique de promotion. Changer de modèle ne touche pas au code.
- **Validation avant entraînement** : colonnes, taille, valeurs de la cible, taux de positifs, plages numériques, taux de manquants, cardinalité. Un CSV corrompu échoue en quelques millisecondes, pas après 5 minutes de GridSearch.
- **Split avant tuning et seuil choisi en out-of-fold** : le test n'est vu ni par le GridSearch ni par le choix du seuil.
- **`model_meta.json` comme contrat entre registre et serving** : l'image API légère n'embarque pas MLflow, mais sait exactement quelle version, quel run et quel seuil elle sert.
- **Alias plutôt que stages** : les stages sont dépréciés depuis MLflow 2.9 ; `champion`/`challenger` sont utilisés, le stage legacy est posé en plus pour la consigne.
- **Lock files** pour des builds reproductibles, ranges lisibles dans les `.txt` pour les humains.
- **`logging` plutôt que `print`**, niveau via `LOG_LEVEL`.
