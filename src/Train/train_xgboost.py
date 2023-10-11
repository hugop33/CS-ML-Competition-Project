from xgboost import XGBRegressor
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from src.Preprocessing import *
from config import *


def load_data(csv_name):
    csv_path = os.path.join(DATA_FOLDER, csv_name)
    return pd.read_csv(csv_path, sep=';')


def main():
    # Chargement des données
    df = load_data(DATA_FILENAME)
    df = date_to_float_col(df, replace=True)
    df["depart_region"] = df["gare_depart"].apply(gare_region)
    df["arrivee_region"] = df["gare_arrivee"].apply(gare_region)
    df["depart_departement"] = df["gare_depart"].apply(gare_departement)
    df["arrivee_departement"] = df["gare_arrivee"].apply(gare_departement)
    # Remove gare_arrivee, gare_depart
    df = df.drop(labels=["gare_arrivee", "gare_depart"], axis=1)
    # One hot encoding
    df = oneHotEncoding(df, "depart_region")
    df = oneHotEncoding(df, "arrivee_region")
    df = oneHotEncoding(df, "depart_departement")
    df = oneHotEncoding(df, "arrivee_departement")
    df = oneHotEncoding(df, "service")

    # df["duree_moyenne"] = df["duree_moyenne"]-df["retard_moyen_arrivee"]x

    xcols = ["duree_moyenne", "nb_train_prevu", "annee", "mois"]

    xcols_to_keep = [c for c in df.columns if c in xcols or c.startswith(
        ("depart_region", "arrivee_region", "depart_departement", "arrivee_departement", "service"))]
    ycols = ["retard_moyen_arrivee"]

    df = df[xcols_to_keep+ycols]

    df_train = df.where(df["annee"] < 2023)
    df_test = df.where(df["annee"] >= 2023)

    y_train = df[ycols]
    X_train = df[xcols_to_keep]

    y_test = df[ycols]
    X_test = df[xcols_to_keep]

    xgb = XGBRegressor(
        n_estimators=1000,
        verbosity=2
    )

    xgb.fit(X_train, y_train)

    score = xgb.score(X_train, y_train)
    print("Trainging score:", score)

    y_pred = xgb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE :", mse)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Boston test and predicted data")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
