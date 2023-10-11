import pandas as pd
import numpy as np

def train_test(df: pd.DataFrame):
    '''
    Prend un dataframe en entrée
    Sépare le dataframe en train [2018 - 2022] et test [2023] set

    Output : X_train, y_train, X_test, y_test
    '''
    assert 'date' in df.columns
    list_months = list(set(df["date"]))
    list_months.sort()

    list_test = list_months[-6:] # 6 derniers mois du dataset i.e données de 2023
    list_train = list_months[:-6]
    df = df.set_index('date')

    train = df.loc[list_train]
    test = df.loc[list_test]  

    y_train = train["retard_moyen_arrivee"].copy()
    X_train = train.drop(["retard_moyen_arrivee"], axis = 1)  

    y_test = test["retard_moyen_arrivee"].copy()
    X_test = test.drop(["retard_moyen_arrivee"], axis = 1)

    return X_train, y_train, X_test, y_test