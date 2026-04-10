Insurance Cost Predictor
Introduction :

Ce projet consiste à concevoir et déployer un système de prédiction du coût d’assurance médicale basé sur des techniques de machine learning. L’objectif est de mettre en œuvre un pipeline complet allant de la préparation des données jusqu’à la mise en production d’un modèle accessible via une API REST.

Le modèle permet d’estimer le coût annuel d’une assurance à partir de plusieurs caractéristiques individuelles, notamment l’âge, le sexe, l’indice de masse corporelle, le statut tabagique, le nombre d’enfants à charge et la région de résidence. Le dataset utilisé provient de Kaggle et contient 1338 observations.

Structure du projet :

Le projet est organisé de manière à séparer clairement les différentes responsabilités.

Le dossier data/ contient le fichier source insurance.csv. Le dossier model/ regroupe les artefacts générés après entraînement, notamment le modèle sérialisé (model.pkl) et les encodeurs (encoders.pkl). Le dossier app/ contient l’API, avec un fichier main.py pour la définition des routes et predict.py pour la logique métier.

À la racine, on trouve le script train.py pour l’entraînement du modèle, le fichier requirements.txt pour les dépendances, ainsi que les fichiers Dockerfile, docker-compose.yml et runtime.txt pour la conteneurisation et le déploiement.

Données et entraînement : 

Le dataset ne contient pas de valeurs manquantes et comprend six variables explicatives ainsi qu’une variable cible représentant le coût de l’assurance.

Les variables numériques sont utilisées telles quelles, tandis que les variables catégorielles sont encodées à l’aide de LabelEncoder. Les encodeurs sont sauvegardés afin de garantir la cohérence entre la phase d’entraînement et la phase de prédiction.

Le modèle retenu est un GradientBoostingRegressor. Ce choix s’explique par sa robustesse sur des données tabulaires et sa capacité à fournir de bonnes performances sans nécessiter de normalisation préalable.

L’évaluation du modèle montre une erreur absolue moyenne d’environ 2483 dollars et un score R² de 0.86 sur l’ensemble de test. Une validation croisée confirme la stabilité du modèle.

API REST  :

L’API est développée avec FastAPI et exposée via Uvicorn. Elle est structurée de manière à charger le modèle une seule fois au démarrage afin d’optimiser les performances.

Deux endpoints principaux sont disponibles. Le premier permet de vérifier que le service est opérationnel. Le second permet d’envoyer des données en entrée et de recevoir une prédiction en sortie.

Les données sont validées avec Pydantic, ce qui permet de garantir que les valeurs respectent des contraintes précises. En cas d’erreur, une réponse explicite est renvoyée.

Une documentation interactive est automatiquement générée et accessible via l’interface Swagger.

Conteneurisation  :

Le projet est conteneurisé avec Docker afin de garantir la reproductibilité de l’environnement d’exécution. L’image Docker inclut l’ensemble du code, des dépendances et du modèle.

Un fichier docker-compose est également fourni pour faciliter le lancement du service. Il permet de configurer le port, les variables d’environnement et les vérifications de santé.

Cette approche simplifie le déploiement sur différents environnements et permet une mise à l’échelle plus facile.

Déploiement  :

Le projet est déployé sur la plateforme Render à partir du dépôt GitHub. La version de Python est fixée via le fichier runtime.txt et les dépendances sont strictement définies dans requirements.txt.

Le service est lancé avec Uvicorn, en utilisant le port fourni dynamiquement par la plateforme. Chaque mise à jour du dépôt déclenche automatiquement un nouveau déploiement.

Problème rencontré  :

Un problème de compatibilité a été rencontré entre certaines versions de scipy et scikit-learn. Cette incompatibilité empêchait le démarrage de l’application.

La solution a consisté à nettoyer l’environnement, installer des versions compatibles des bibliothèques, puis réentraîner le modèle et régénérer les fichiers sérialisés. Les versions ont ensuite été figées dans le fichier requirements afin d’éviter toute régression.

Dépendances  :

Le projet repose sur FastAPI, Uvicorn, Pydantic, scikit-learn, pandas, numpy et scipy. La version de Python utilisée est la 3.11.8.

Conclusion  :

Ce projet illustre un cas concret de mise en production d’un modèle de machine learning. Il couvre l’ensemble des étapes nécessaires, depuis la préparation des données jusqu’au déploiement d’une API fonctionnelle.

Il met en évidence l’importance de la structuration du projet, de la gestion des dépendances et de la reproductibilité des environnements dans un contexte professionnel.
