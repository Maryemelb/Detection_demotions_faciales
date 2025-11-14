
```markdown
# 🤖 API de Détection d'Émotions Faciales

Ce projet est un prototype d'API d'analyse émotionnelle. Son objectif est de détecter un visage dans une image fournie, de prédire l'émotion de ce visage à l'aide d'un modèle de Deep Learning (CNN), et d'enregistrer le résultat dans une base de données PostgreSQL.

Ce prototype sert à valider la faisabilité technique d'un futur produit SaaS destiné à l'analyse de réactions utilisateurs (UX, tests produits).

## ✨ Fonctionnalités

* **Détection de Visage** : Utilise OpenCV et le classifieur Haar Cascade pour localiser automatiquement les visages dans une image.
* **Prédiction d'Émotion** : Emploie un modèle de réseau de neurones convolutif (CNN) entraîné avec TensorFlow/Keras pour classifier l'émotion (ex: joie, tristesse, colère, surprise).
* **API RESTful** : Une API FastAPI expose deux points de terminaison :
    * `POST /predict_emotion` : Reçoit une image, effectue la détection et la prédiction, et sauvegarde le résultat.
    * `GET /history` : Renvoie l'historique de toutes les prédictions stockées.
* **Persistance des Données** : Chaque prédiction réussie est enregistrée dans une base de données PostgreSQL via SQLAlchemy.

## 🛠️ Stack Technique

* **Python 3.10+**
* **Modèle IA** : TensorFlow / Keras (pour le CNN), OpenCV (pour Haar Cascade)
* **API** : FastAPI
* **Base de Données** : PostgreSQL
* **ORM** : SQLAlchemy
* **Tests** : Pytest
* **CI/CD** : GitHub Actions

## 📂 Structure du Projet

```


├── .github/workflows/

│                 └── demo.yml           \# Workflow GitHub Actions pour les tests

├── dataset/

│         └── test/
      
│         └── train/
├── images/        \# Dossier où les images testées ont été enregistrées

├── pipeline/

│        └── detect_and_predict.py       \# contient la fonction d’entraînement et de prédiction

├── tests/

│      └── test_model_prediction.py        \# Tests unitaires

├── .env           \# Fichier d'exemple pour les variables d'environnement

├── main.py                \# Fichier principal de l'API FastAPI

├── requirements.txt       \# Dépendances Python

├── best_model.keras       \# Le modèle CNN entraîné

└── README.md              \# documentation

````

## 🚀 Installation et Lancement

Suivez ces étapes pour configurer et lancer le projet localement.

### 1. Prérequis

* Python 3.10 ou supérieur
* Un serveur PostgreSQL en cours d'exécution

### 2. Cloner le Dépôt

```bash
git clone <url-de-votre-depot>
cd <nom-du-depot>
````

### 3\. Configurer l'Environnement

Créez et activez un environnement virtuel :

```bash
python -m venv venv
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

### 4\. Configurer la Base de Données

Créez un fichier `.env` à la racine du projet par exemple:

**Fichier `.env` :**

```ini
# 
DATABASE_NAME= "exe_name"
DATABASE_PASSWORD="exe_pass"
DATABASE_PORT=5432
DATABASE_HOST="localhost"
DATABASE_USER= "exe_password"

```

### 5\. Lancer l'API

Utilisez `uvicorn` pour démarrer le serveur FastAPI :

```bash
uvicorn main:app --reload
```

L'API est maintenant accessible à l'adresse `http://127.0.0.1:8000`. La documentation interactive (Swagger UI) est disponible sur `http://127.0.0.1:8000/docs`.

## 📈 Utilisation de l'API

### `POST /predict_emotion`

Ce point de terminaison permet de soumettre une image pour analyse.

**Exemple avec `curl` :**

```bash
 visitez ce lien: 
 [http://127.0.0.1:8000/predict_emotion]
```

**Réponse Attendue (Succès) :**

```json
{
  "emotion": "happy",
  "confidence": 0.92,
}
```

### `GET /history`

Ce point de terminaison renvoie la liste de toutes les prédictions enregistrées.


**Réponse Attendue :**

```json
[
  {
    "id": 1,
    "emotion": "happy",
    "confidence": 0.92,
    "created_at": "2025-11-14T15:30:00Z"
  },
  {
    "id": 2,
    "emotion": "surprised",
    "confidence": 0.78,
    "created_at": "2025-11-14T15:31:12Z"
  }
]
```

## 🧩 Composants Clés

### 1\. Entraînement du Modèle (`Emotion_CNN_Training.ipynb`)

Le notebook Jupyter détaille les étapes de :

  * Chargement des données avec `tf.keras.utils.image_dataset_from_directory`.
  * Prétraitement (normalisation, augmentation des données).
  * Construction du modèle CNN (Conv2D, MaxPooling2D, Dropout, Dense).
  * Entraînement (`adam`, `categorical_crossentropy`).
  * Évaluation et sauvegarde du modèle avec keras.

### 2\. Script de Test (`detect_and_predict.py`)

Ce script permet de tester le pipeline complet (OpenCV + Keras) sur une seule image sans démarrer l'API.

### 3\. Tests et CI/CD

Les tests unitaires vérifient :

  * Le chargement correct du modèle.
  * Le format de la réponse de prédiction.

Le workflow GitHub Actions (défini dans `.github/workflows/demo.yml`) exécute ces tests automatiquement à chaque `push` ou `pull_request` sur les branches `main` et `develop`, en utilisant un service PostgreSQL pour l'intégration.

Pour lancer les tests localement :

```bash
pytest -v
```
