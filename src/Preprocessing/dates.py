import numpy as np
import pandas as pd


def date_to_float_col(df: pd.DataFrame, replace=False):
    '''
    Prend en entrée un dataframe contenant une colonne 'date' au format 'yyyy-mm'
    Ajoute deux colonnes 'annee' et 'mois' contenant les valeurs correspondantes et converties en float
    Si replace est choisi à True, supprime la colonne 'date'

    Args:
    -----
        `df` (pd.DataFrame): dataframe contenant une colonne 'date' au format 'yyyy-mm'
        `replace` (bool): si True, supprime la colonne 'date' du dataframe

    Returns:
    --------
        `df` (pd.DataFrame): dataframe avec deux colonnes 'annee' et 'mois' au format float
    '''
    assert 'date' in df.columns

    date = df["date"].copy()
    date = date.str.split('-')

    list_year = [date[k][0] for k in range(len(date))]
    list_month = [date[k][1] for k in range(len(date))]

    df_date = pd.DataFrame({
        'annee': list_year,
        'mois': list_month,
    }).astype(float)

    df = pd.concat([df, df_date], axis=1)
    if replace:
        df = df.drop(labels=['date'], axis=1)

    return df
