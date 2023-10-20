# Competition TGV

## Introduction

Le projet "Competition TGV" vise à prédire les retards de train à la SNCF et leurs causes.

## Structure du projet

### Data_Visualisation
Contient des scripts ou des notebooks pour la visualisation des données.

* **data_visualisation.py** : 
  - Ce script se concentre sur la visualisation des données associées aux annulations de TGV.
  - Il contient une fonction `nombre_annulations_mensuel` qui prend en entrée un fichier CSV. Cette fonction lit le fichier pour obtenir un dataframe, transforme les dates en valeurs numériques flottantes, calcule le nombre moyen d'annulations par mois pour chaque année, et trace un graphique montrant ces moyennes.

### Preprocessing
Contient des scripts ou des fonctions pour la préparation et la transformation des données avant l'entraînement.

* **dates.py** :
  - Contient des fonctions liées à la manipulation et à la conversion de dates.
  - La fonction `date_to_float_col` convertit les dates au format 'yyyy-mm' en deux colonnes flottantes, 'annee' et 'mois'.
  
* **history_inference.py** :
  - Contient des fonctions pour inférer des informations basées sur l'historique des données.
  - La fonction `infer_annulations` utilise l'historique pour estimer le nombre d'annulations pour un mois donné.

* **lieux_gares.py** :
  - Contient un dictionnaire avec des informations sur différentes gares, notamment leur région et leur département.

* **one_hot_encoding.py** :
  - Contient une fonction pour effectuer du "one hot encoding" sur un dataframe pour une colonne spécifique.
  
* **pipeline.py** :
  - Contient des fonctions qui composent le pipeline de prétraitement des données.
  - La fonction `scale` met à l'échelle les données en utilisant soit `MinMaxScaler` soit `StandardScaler`.

* **train_test_split.py** :
  - Contient des fonctions pour diviser un dataframe en ensembles d'entraînement et de test.
  - La fonction `train_test` divise les données en fonction des années, en utilisant les données de 2018 à 2022 pour l'entraînement et les données de 2023 pour le test.

### Train
Dossier destiné à l'entraînement des modèles.

* **train_lasso.py** :
  - Se concentre sur l'entraînement d'un modèle de régression Lasso.
  - La fonction principale prépare les données et entraîne un modèle Lasso sur les données d'entraînement.

* **train_svm.py** :
  - Se concentre sur l'entraînement d'un modèle de Support Vector Regression.
  - La fonction principale prépare les données et entraîne un modèle SVR sur les données d'entraînement.
  
* **train_lgbm.py** :
  - Se concentre sur l'entraînement d'un modèle LightGBM.
  - La fonction principale prépare les données et entraîne un modèle LightGBM sur les données d'entraînement.

* **train_nn.py** :
  - Se concentre sur l'entraînement d'un réseau de neurones en utilisant PyTorch.
  - Contient une classe `TGVDataset` pour faciliter le chargement et la manipulation des données.

* **train_random_forest.py** :
  - Se concentre sur l'entraînement d'un modèle de forêt aléatoire.
  - La fonction principale instancie et entraîne un modèle de forêt aléatoire sur les données d'entraînement.

* **train_xgboost.py** :
  - Se concentre sur l'entraînement d'un modèle XGBoost.
  - La fonction principale instancie et entraîne un modèle XGBoost sur les données d'entraînement, avec un arrêt prématuré basé sur la performance de validation.

### Evaluation
Contient des scripts ou des fonctions pour évaluer les performances des modèles entraînés.

* **eval_lasso.py** :
  - Actuellement, ce script est vide. Il est prévu pour contenir des fonctions ou des procédures d'évaluation associées à un modèle Lasso.

* **eval_xgboost.py** :
  - Actuellement, ce script est vide. Il est prévu pour contenir des fonctions ou des procédures d'évaluation associées à un modèle XGBoost.

## Installation et dépendances

1. Clonez ce dépôt dans votre environnement local.
2. Installez les dépendances nécessaires en exécutant la commande suivante :
   ```bash
   pip install -r requirements.txt

