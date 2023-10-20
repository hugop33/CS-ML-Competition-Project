import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import *


def plot_correlation(file_path: str):
    """
    Cette fonction permet de visualiser la matrice de corrélation des variables numériques et des retards moyens entre différents trajets.

    Args:
    -----
        `file_path` (str): chemin vers le fichier csv contenant les données.
    """

    data = pd.read_csv(file_path, sep=';')
    correlation_matrix = data.corr()

    # Configurer la taille de la figure
    plt.figure(figsize=(12, 10))

    # Créer une heatmap
    sns.heatmap(correlation_matrix, annot=True,
                cmap='coolwarm', linewidths=.5, fmt=".2f")

    # Titre et labels
    plt.title('Matrice de Corrélation des Variables Numériques', fontsize=15)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)

    # Montrer le plot
    plt.tight_layout()

    # Créer une nouvelle colonne qui combine la gare de départ et d'arrivée
    data['traject'] = data['gare_depart'] + ' - ' + data['gare_arrivee']

    # Pivoter les données pour avoir une colonne par trajet, avec le retard moyen comme valeur
    pivot_data = data.pivot(index='date', columns='traject',
                            values='retard_moyen_arrivee')

    # Calculer la matrice de corrélation pour les nouvelles colonnes
    correlation_matrix_traject = pivot_data.corr()

    # Afficher les dimensions de la matrice de corrélation pour vérifier la faisabilité de la visualisation
    correlation_matrix_traject.shape

    # Configurer la taille de la figure
    plt.figure(figsize=(20, 18))

    # Créer une heatmap
    sns.heatmap(correlation_matrix_traject, cmap='coolwarm', linewidths=.5)

    # Titre et labels
    plt.title(
        'Matrice de Corrélation des Retards Moyens entre Différents Trajets', fontsize=15)
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8, rotation=0)

    # Montrer le plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = DATA_PATH
    plot_correlation(file_path)
