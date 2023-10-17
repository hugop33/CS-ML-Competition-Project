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
def top_gares_retard(fichier_csv):
    df = pd.read_csv(fichier_csv, sep=";")
    df_date = date_to_float_col(df, replace=False)
    annees = df_date['annee'].unique()
    plt.figure(figsize=(12, 8))
    for annee in annees:
        subset = df_date[df_date['annee'] == annee]
        total_retard_by_gare = subset.groupby('gare_depart')['retard_moyen_tous_trains_arrivee'].sum()
        top_5_gares = total_retard_by_gare.sort_values(ascending=False).head(5)
        
        plt.bar(top_5_gares.index + f" ({annee})", top_5_gares.values, label=str(annee))
        
    # 4. Affichage des résultats
    plt.title("Top 5 des gares avec le plus de retard par année")
    plt.xlabel('Gares')
    plt.ylabel('Total de retard (minutes, heures, etc. en fonction de l\'unité de mesure)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def ratio_retard_par_train(fichier_csv):
    # 1. Lire le fichier CSV
    df = pd.read_csv(fichier_csv, sep=";")
    
    # 2. Convertir les dates si nécessaire (en supposant que vous avez une fonction date_to_float_col)
    df_date = date_to_float_col(df, replace=False)
    annees = df_date['annee'].unique()
    
    plt.figure(figsize=(12, 8))
    
    # 3. Pour chaque année, trouvez le top 5 des gares basé sur le ratio retard par train prévu
    for annee in annees:
        subset = df_date[df_date['annee'] == annee]
        total_retard_by_gare = subset.groupby('gare_depart')['retard_moyen_tous_trains_arrivee'].sum()
        total_trains_by_gare = subset.groupby('gare_depart')['nb_train_prevu'].sum()
        
        ratio_retard = total_retard_by_gare / total_trains_by_gare
        top_5_gares = ratio_retard.sort_values(ascending=False).head(5)
        
        plt.bar(top_5_gares.index + f" ({annee})", top_5_gares.values, label=str(annee))
        
    # 4. Affichage des résultats
    plt.title("Top 5 des gares basé sur le ratio de retard par train prévu par année")
    plt.xlabel('Gares')
    plt.ylabel('Ratio de retard par train prévu')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    nombre_annulations_mensuel("C:/Users/Dan/Downloads/aregularite-mensuelle-tgv-aqst.csv")
    ratio_annulations_mensuel("C:/Users/Dan/Downloads/aregularite-mensuelle-tgv-aqst.csv")
    top_gares_retard("C:/Users/Dan/Downloads/aregularite-mensuelle-tgv-aqst.csv")
    ratio_retard_par_train("C:/Users/Dan/Downloads/aregularite-mensuelle-tgv-aqst.csv")