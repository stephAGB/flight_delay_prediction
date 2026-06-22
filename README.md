# Flight Delay Prediction - Architecture Classique ML

Ce projet implémente un pipeline de prédiction de retards de vols avec une architecture de projet Machine Learning classique et modulaire.

## Structure du Projet

```text
flight_delay_prediction/
├── data/                    # Répertoire pour les données
│   ├── raw/                 # Données brutes
│   └── processed/           # Données préparées pour l'entraînement
├── artifacts/               # Modèles entraînés et objets sérialisés
├── src/                     # Code source
│   ├── __init__.py
│   ├── preprocessing/       # Scripts de traitement
│   │   ├── __init__.py
│   │   └── reduce_data.py   # Nettoyage et filtrage des données
│   ├── train.py             # Entraînement du modèle
│   └── api.py               # API FastAPI pour les prédictions
├── tests/                   # Tests unitaires et d'intégration
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_api.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

1. Créer et activer un environnement virtuel :
   ```bash
   python -m venv .venv
   # Windows :
   .venv\Scripts\activate
   # Linux/macOS :
   source .venv/bin/activate
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### 1. Prétraitement / Réduction des données
Placez vos fichiers bruts dans `data/raw/` puis lancez :
```bash
python src/preprocessing/reduce_data.py
```

### 2. Entraînement du modèle
Lancez l'entraînement à partir des données pré-traitées :
```bash
python src/train.py
```
Le modèle entraîné sera sauvegardé dans `artifacts/model.joblib`.

### 3. Service de prédiction (API)
Démarrez l'API FastAPI localement :
```bash
uvicorn src.api:app --reload
```
Accédez à l'interface Swagger pour tester les requêtes à l'adresse suivante : http://127.0.0.1:8000/docs.

### 4. Tests
Exécutez la suite de tests automatisés :
```bash
pytest tests/
```

## Gestion des données (DVC & DAGsHub)

Les données volumineuses et les modèles du projet ne sont pas versionnés sur Git. Ils sont suivis par DVC et hébergés sur le stockage distant de **DAGsHub**.

### Récupérer les données (Collaborateurs)

1. **Si le dépôt est Public** :
   Exécutez simplement la commande suivante à la racine :
   ```bash
   dvc pull
   ```

2. **Si le dépôt est Privé** :
   Vous devez d'abord être membre du dépôt sur DAGsHub et configurer vos accès personnels en local (dans votre config locale DVC qui n'est pas suivie par Git) :
   ```bash
   dvc remote modify origin-dvc --local auth basic
   dvc remote modify origin-dvc --local user VOTRE_PSEUDO_DAGSHUB
   dvc remote modify origin-dvc --local password VOTRE_TOKEN_DAGSHUB
   ```
   Puis téléchargez les fichiers :
   ```bash
   dvc pull
   ```

### Pousser des données (Pour mettre à jour le stockage)
Si vous ajoutez ou modifiez des fichiers de données locaux suivis par DVC (par exemple avec `dvc add data/raw/flights.csv`), poussez les fichiers physiques sur DAGsHub avec :
```bash
dvc push
```
*(Vos identifiants locaux configurés ci-dessus seront utilisés pour l'autorisation).*

