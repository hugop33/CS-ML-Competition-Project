import pandas as pd
import numpy as np


def oneHotEncoding(df: pd.DataFrame, col_name: str):
    '''
    Pour un dataframe et un nom de colonne donnés:
    - Supprime la colonne du dataframe
    - Effectue un one hot encoding de la colonne 'col_name' et l'ajoute au dataframe
    - Le nom de chaque nouvelle colonne est 'col_name' + '_valeur'

    Args:
    -----
        `df` (pd.DataFrame): dataframe contenant la colonne à encoder
        `col_name` (str): nom de la colonne à encoder

    Returns:
    --------
        `df` (pd.DataFrame): dataframe avec la colonne encodée
    '''
    new_cols = pd.get_dummies(df[[col_name]])
    df = df.drop(labels=[col_name], axis=1)
    df = pd.concat([df, new_cols], axis=1)
    return df
