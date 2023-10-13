import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from config import *
from .dates import date_to_float_col
from .train_test_split import train_test
from .one_hot_encoding import oneHotEncoding
from .lieux_gares import *


def scale(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, scaler: str = "minmax"):
    if scaler == "minmax":
        Xscaler = MinMaxScaler()
        yscaler = MinMaxScaler()
    elif scaler == "standard":
        Xscaler = StandardScaler()
        yscaler = StandardScaler()
    X_train = Xscaler.fit_transform(X_train)
    X_test = Xscaler.transform(X_test)
    y_train, y_test = y_train.values.reshape(-1,
                                             1), y_test.values.reshape(-1, 1)
    y_train = yscaler.fit_transform(y_train)
    y_test = yscaler.transform(y_test)
    return X_train, y_train, X_test, y_test


def data_pipeline(csv_name, scaler="minmax"):
    csv_path = os.path.join(DATA_FOLDER, csv_name)
    df = pd.read_csv(csv_path, sep=';')
    df = date_to_float_col(df, replace=False)
    df["depart_region"] = df["gare_depart"].apply(gare_region)
    df["arrivee_region"] = df["gare_arrivee"].apply(gare_region)
    df["depart_departement"] = df["gare_depart"].apply(gare_departement)
    df["arrivee_departement"] = df["gare_arrivee"].apply(gare_departement)

    df["distances_km"] = df.apply(
        lambda row: distance_intergares(row["gare_depart"], row["gare_arrivee"]), axis=1)
    # Remove gare_arrivee, gare_depart
    df = df.drop(labels=["gare_arrivee", "gare_depart"], axis=1)
    # One hot encoding
    df = oneHotEncoding(df, "depart_region")
    df = oneHotEncoding(df, "arrivee_region")
    df = oneHotEncoding(df, "depart_departement")
    df = oneHotEncoding(df, "arrivee_departement")
    df = oneHotEncoding(df, "service")

    # df["duree_moyenne"] = df["duree_moyenne"]-df["retard_moyen_arrivee"]x

    xcols = ["duree_moyenne", "nb_train_prevu",
             "annee", "mois", "date", "distances_km"]

    xcols_to_keep = [c for c in df.columns if c in xcols or c.startswith(
        ("depart_region", "arrivee_region", "depart_departement", "arrivee_departement", "service"))]
    ycols = ["retard_moyen_arrivee"]

    df = df[xcols_to_keep+ycols]

    X_train, y_train, X_test, y_test = train_test(df)
    scaled = scale(X_train, y_train, X_test, y_test, scaler=scaler)

    return scaled
