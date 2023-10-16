from src.Preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt

def nombre_annulations_mensuel(fichier_csv):
    df = pd.read_csv(fichier_csv, sep=";")
    df_date = date_to_float_col(df, replace=False)
    annees = df_date['annee'].unique() 
    moyennes_par_annee = {}
    # 3. Pour chaque année, calcule le nombre moyen d'annulations par mois
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

    # 3. Pour chaque année, calcule le ratio d'annulation par mois
def ratio_annulations_mensuel(fichier_csv):
    df = pd.read_csv(fichier_csv, sep=";")
    df_date = date_to_float_col(df, replace=False)
    annees = df_date['annee'].unique() 
    ratios_par_annee = {}
    for annee in annees:
        subset = df_date[df_date['annee'] == annee]
        total_annulations = subset.groupby('mois')['nb_annulation'].sum()
        total_trains_prevus = subset.groupby('mois')['nb_train_prevu'].sum()
        ratio_mois = total_annulations / total_trains_prevus
        ratios_par_annee[annee] = ratio_mois

    # 4. Tracer ces ratios pour chaque année
    for annee, ratio_mois in ratios_par_annee.items():
        plt.plot(ratio_mois.index, ratio_mois.values, label=str(annee))

    plt.title("Ratio d'annulations par mois")
    plt.xlabel('Mois')
    plt.ylabel('Ratio d\'annulations')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    ratio_annulations_mensuel("C:/Users/Dan/Downloads/aregularite-mensuelle-tgv-aqst.csv")
