import pandas as pd
import numpy as np

def oneHotEncoding(df: pd.DataFrame, col_name: str):
    '''
    Pour un dataframe et un nom de colonne donnés:
    - Supprime la colonne du dataframe
    - Effectue un one hot encoding de la colonne 'col_name' et l'ajoute au dataframe
    - Le nom de chaque nouvelle colonne est 'col_name' + '_valeur'
    '''
    new_cols = pd.get_dummies(df[[col_name]])
    df = df.drop(labels=[col_name], axis = 1)
    df = pd.concat([df, new_cols], axis = 1)
    return df