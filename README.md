<p align="center">
  <img src="outputs/images/at&t_logo.png" alt="Logo AT&T" width="220">
</p>

<h1 align="center">AT&T Spam Detector</h1>

<p align="center">
  Projet Deep Learning — Bloc 4 — Certification RNCP CDSD — Jedha
</p>

## Description du projet

Ce dépôt présente la conception et l'évaluation d'un détecteur de SMS indésirables pour un contexte opérateur télécom. Le travail couvre l'analyse exploratoire du jeu de données, la construction de modèles PyTorch entraînés from scratch, puis l'affinage d'un modèle de transfer learning basé sur DistilBERT.

L'approche suit une progression volontairement pédagogique : partir d'un modèle dense simple, explorer des architectures séquentielles, puis exploiter un transformeur pré-entraîné pour maximiser les performances sur un corpus de taille modeste.

## Objectifs

- Comprendre la distribution des messages ham et spam, leurs longueurs et leurs signaux lexicaux.
- Mettre en place un pipeline de prétraitement, de tokenisation et d'évaluation reproductible.
- Comparer plusieurs familles de modèles de classification texte.
- Identifier le meilleur compromis entre performance, complexité et exploitabilité pour un usage type AT&T.
- Produire des livrables exploitables pour un rapport, une soutenance ou une relecture technique.

## Livrables

| Livrable | Emplacement |
| --- | --- |
| Notebook EDA et baseline | `notebooks/01_eda_baseline.ipynb` |
| Notebook modèles séquentiels | `notebooks/02_sequential_models.ipynb` |
| Notebook transfer learning DistilBERT | `notebooks/03_transfer_learning_models.ipynb` |
| Jeu de données SMS | `data/spam.csv` |
| Figures exportées | `outputs/images/` |
| Poids des modèles entraînés | `outputs/models/` |
| Dépendances projet | `pyproject.toml`, `requirements.txt` |
| Point d'entrée Python minimal | `main.py` |

Les figures générées par les notebooks incluent notamment la distribution des classes, les analyses de longueur de texte, les mots fréquents, les indicateurs de spam, les courbes d'entraînement, les matrices de confusion, les courbes ROC et, pour DistilBERT, la courbe précision-rappel et l'histogramme de confiance.

## Jeu de données

Le projet s'appuie sur le jeu **SMS Spam Collection** : 5 572 SMS étiquetés `ham` ou `spam`, avec un déséquilibre marqué en faveur des messages légitimes.

- Fichier local : `data/spam.csv`
- Encodage utilisé dans les notebooks : `latin-1`
- Colonnes exploitées : `v1` pour le label, `v2` pour le texte du message
- Découpage : 70 % entraînement, 15 % validation, 15 % test, avec stratification sur la classe spam

## Structure du projet

```text
projet-spam-detector-bloc4/
├── data/
│   └── spam.csv
├── notebooks/
│   ├── 01_eda_baseline.ipynb
│   ├── 02_sequential_models.ipynb
│   └── 03_transfer_learning_models.ipynb
├── outputs/
│   ├── images/
│   └── models/
├── .env.dist
├── main.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## Modèles

| Modèle | Notebook | Artefact local |
| --- | --- | --- |
| Baseline dense : embedding + pooling + couche dense | `01_eda_baseline.ipynb` | `outputs/models/best_baseline.pth` |
| LSTM simple | `02_sequential_models.ipynb` | `outputs/models/best_simple_lstm.pth` |
| BiLSTM | `02_sequential_models.ipynb` | `outputs/models/best_bilstm.pth` |
| Conv1D + BiLSTM | `02_sequential_models.ipynb` | `outputs/models/best_conv1d+bilstm.pth` |
| DistilBERT fine-tuné | `03_transfer_learning_models.ipynb` | `outputs/models/best_distilbert_spam.pth` |
| Export Hugging Face DistilBERT | `03_transfer_learning_models.ipynb` | `outputs/models/spam_detector_distilbert/` |

Les petits checkpoints PyTorch peuvent être régénérés localement en réexécutant les notebooks correspondants. Les artefacts DistilBERT sont volumineux et ne sont pas versionnés sur GitHub.

## Prérequis

- Python 3.12 ou version ultérieure
- `pip` pour l'installation des dépendances
- Un environnement virtuel recommandé
- Jupyter ou JupyterLab pour exécuter les notebooks
- Espace disque suffisant pour PyTorch, Transformers et les exports de modèles
- GPU optionnel mais utile pour l'entraînement DistilBERT
- Token Hugging Face optionnel pour accélérer le téléchargement du modèle de base ; copier `.env.dist` vers `.env` et renseigner `HF_TOKEN` si besoin

## Installation

### Avec pip et `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Le fichier `requirements.txt` est généré à partir des dépendances du projet. Après modification de `pyproject.toml`, vous pouvez le régénérer avec :

```bash
uv export --format requirements-txt --no-hashes -o requirements.txt
```

### Alternative avec `uv`

```bash
uv sync
```

Les dépendances principales du projet sont : `torch`, `transformers`, `scikit-learn`, `matplotlib`, `seaborn`, `spacy`, `python-dotenv` et `datasets`.

## Exécution

1. Installer les dépendances.
2. Vérifier la présence de `data/spam.csv`.
3. Lancer Jupyter depuis la racine du dépôt.
4. Exécuter les notebooks dans l'ordre :
   - `notebooks/01_eda_baseline.ipynb`
   - `notebooks/02_sequential_models.ipynb`
   - `notebooks/03_transfer_learning_models.ipynb`

Chaque notebook crée le dossier `outputs/images/` si nécessaire et enregistre ses figures dans ce répertoire. Les checkpoints sont écrits dans `outputs/models/`.

### Modèles absents du dépôt GitHub

Le dépôt distant exclut volontairement le contenu de `outputs/models/` à l'exception de `outputs/models/.gitkeep`, car les poids DistilBERT dépassent la limite de taille imposée par GitHub. Après un clone du dépôt, deux options sont possibles :

- Réentraîner les modèles en réexécutant les notebooks, en commençant par `03_transfer_learning_models.ipynb` si seul DistilBERT est nécessaire.
- Copier localement les artefacts déjà produits dans `outputs/models/` si vous disposez d'une archive ou d'un lien de partage fourni en complément du dépôt.

Les notebooks restent la source de vérité pour reproduire les métriques, les graphiques et les sauvegardes de modèles.

## Auteur et contexte

**RANJAKASOA Raphaël Marcellin**

Projet réalisé dans le cadre du **Bloc 4 — Deep Learning** de la certification **RNCP CDSD** à **Jedha**, autour d'un cas d'usage de détection de spam SMS pour **AT&T**.
