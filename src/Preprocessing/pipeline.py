import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from config import *
from .dates import date_to_float_col
from .train_test_split import train_test
from .one_hot_encoding import oneHotEncoding
from .lieux_gares import *
from .history_inference import infer_annulations


def scale(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, scaler: str = "minmax"):
    """
    Scales the data using either MinMaxScaler or StandardScaler

    Input:
        - X_train: training data
        - y_train: training target
        - X_test: testing data
        - y_test: testing target
        - scaler: either "minmax" or "standard"
    Output:
        - X_train: scaled training data, pd.DataFrame
        - y_train: scaled training target, pd.DataFrame
        - X_test: scaled testing data, pd.DataFrame
        - y_test: scaled testing target, pd.DataFrame
    """
    if scaler == "minmax":
        Xscaler = MinMaxScaler()
        yscaler = MinMaxScaler()
    elif scaler == "standard":
        Xscaler = StandardScaler()
        yscaler = StandardScaler()
    X_cols, y_cols = X_train.columns, y_train.columns
    X_train = Xscaler.fit_transform(X_train)
    X_test = Xscaler.transform(X_test)
    y_train, y_test = y_train.values.reshape(-1,
                                             1), y_test.values.reshape(-1, 1)
    y_train = yscaler.fit_transform(y_train)
    y_test = yscaler.transform(y_test)

    X_train = pd.DataFrame(X_train, columns=X_cols)
    X_test = pd.DataFrame(X_test, columns=X_cols)
    y_train = pd.DataFrame(y_train, columns=y_cols)
    y_test = pd.DataFrame(y_test, columns=y_cols)
    return X_train, y_train, X_test, y_test


def data_pipeline_1(csv_name, scaler="minmax"):
    csv_path = os.path.join(DATA_FOLDER, csv_name)
    df = pd.read_csv(csv_path, sep=';')
    df = date_to_float_col(df, replace=False)
    df = infer_annulations(df)
    df["depart_region"] = df["gare_depart"].apply(gare_region)
    df["arrivee_region"] = df["gare_arrivee"].apply(gare_region)
    df["depart_departement"] = df["gare_depart"].apply(gare_departement)
    df["arrivee_departement"] = df["gare_arrivee"].apply(gare_departement)

    df["cities"] = df["gare_depart"] + df["gare_arrivee"].apply(
        lambda x: "|" + x)
    distance_mapping = distance_map(STATIONS_PATH, df["cities"].unique())
    df["distances_km"] = df["cities"].apply(
        lambda x: distance_mapping[(x.split("|")[0], x.split("|")[1])])
    # Remove gare_arrivee, gare_depart
    df = df.drop(labels=["gare_arrivee", "gare_depart", "cities"], axis=1)
    # One hot encoding
    df = oneHotEncoding(df, "depart_region")
    df = oneHotEncoding(df, "arrivee_region")
    df = oneHotEncoding(df, "depart_departement")
    df = oneHotEncoding(df, "arrivee_departement")
    df = oneHotEncoding(df, "service")

    # df["duree_moyenne"] = df["duree_moyenne"]-df["retard_moyen_arrivee"]x

    xcols = ["duree_moyenne", "nb_train_prevu",
             "annee", "mois", "date", "distances_km", "annulations_mois"]

    xcols_to_keep = [c for c in df.columns if c in xcols or c.startswith(
        ("depart_region", "arrivee_region", "depart_departement", "arrivee_departement", "service"))]
    ycols = ["retard_moyen_arrivee"]

    df = df[xcols_to_keep+ycols]

    X_train, y_train, X_test, y_test = train_test(df)
    scaled = scale(X_train, y_train, X_test, y_test, scaler=scaler)

    return scaled


def data_pipeline_2(csv_name, scaler="minmax"):
    csv_path = os.path.join(DATA_FOLDER, csv_name)
    df = pd.read_csv(csv_path, sep=';')
    X_train, y_train, X_test, y_test = data_pipeline_1(csv_name,scaler)
    X_train["retard_predit"] = X_train.apply(lasso)
    X_test["retard_predit"] = X_test.apply(lasso)
    y_train,y_test=train_test2(df)
    scaled = scale(X_train, y_train, X_test, y_test, scaler=scaler)
    return scaled


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_pipeline_1(DATA_FILENAME)
    print("training shape", X_train.shape)
