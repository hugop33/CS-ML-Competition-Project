from xgboost import XGBRegressor
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from src.Preprocessing import *
from config import *


def load_data(csv_name):
    csv_path = os.path.join(DATA_FOLDER, csv_name)
    df = pd.read_csv(csv_path, sep=';')
    df = date_to_float_col(df, replace=False)
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

    xcols = ["duree_moyenne", "nb_train_prevu", "annee", "mois", "date"]

    xcols_to_keep = [c for c in df.columns if c in xcols or c.startswith(
        ("depart_region", "arrivee_region", "depart_departement", "arrivee_departement", "service"))]
    ycols = ["retard_moyen_arrivee"]

    df = df[xcols_to_keep+ycols]

    X_train, y_train, X_test, y_test = train_test(df)

    return X_train, y_train, X_test, y_test


def train(X_train, y_train, X_test, y_test, **kwargs):
    xgb = XGBRegressor(
        **kwargs
    )

    xgb.fit(X_train, y_train, early_stopping_rounds=100,
            eval_set=[(X_train, y_train), (X_test, y_test)
                      ],
            )
    score = xgb.score(X_train, y_train)
    print("Training score :", score)
    return xgb


def plot_test(xgb, X_test, y_test):
    y_pred = xgb.predict(X_test)
    score = xgb.score(X_test, y_test)
    print("Test score :", score)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE :", mse)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Boston test and predicted data")
    plt.legend()
    plt.show()


def grid_search(X_train, y_train, **grid):
    gs_cv = GridSearchCV(
        XGBRegressor(), grid, cv=5, refit=True, verbose=2
    )
    gs_cv.fit(X_train, y_train)
    print(gs_cv.best_params_)
    best_model = gs_cv.best_estimator_
    return best_model


def main():
    # Chargement des données
    X_train, y_train, X_test, y_test = load_data(DATA_FILENAME)

    xgb = train(X_train, y_train, X_test, y_test,
                n_estimators=10000,
                max_depth=3,
                learning_rate=0.01,
                verbosity=2
                )
    # grid = {"n_estimators": [500, 1000, 1500],
    #         "max_depth": [1, 3, 5, 10],
    #         "learning_rate": [0.01],
    #         "verbosity": [1]}
    # xgb = grid_search(X_train, y_train,
    #                   **grid
    #                   )

    plot_test(xgb, X_test, y_test)


if __name__ == "__main__":
    main()
