from src.Preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt

def nombre_annulations_mensuel(fichier_csv):
    # 1. Charger le fichier CSV
    df = pd.read_csv(fichier_csv, sep=";")
    df_date = date_to_float_col(df, replace=False)

    # 2. Extraire les années uniques
    annees = df_date['annee'].unique()  # Assurez-vous d'avoir des années uniques

    # Initialiser un dictionnaire pour stocker les moyennes par mois pour chaque année
    moyennes_par_annee = {}

    # 3. Pour chaque année, calculer le nombre moyen d'annulations par mois
    for annee in annees:
        subset = df_date[df_date['annee'] == annee]
        moyenne_mois = subset.groupby('mois')['nb_annulation'].mean()
        moyennes_par_annee[annee] = moyenne_mois

    # 4. Tracer ces moyennes pour chaque année
    for annee, moyenne_mois in moyennes_par_annee.items():
        plt.plot(moyenne_mois.index, moyenne_mois.values, label=str(annee))

    plt.title("Nombre moyen d'annulations par mois")
    plt.xlabel('Mois')
    plt.ylabel('Nombre moyen annulations')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    nombre_annulations_mensuel("C:/Users/Dan/Downloads/aregularite-mensuelle-tgv-aqst.csv")
