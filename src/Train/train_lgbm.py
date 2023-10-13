from lightgbm import LGBMRegressor, early_stopping
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from src.Preprocessing import *
from config import *


def scale(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    Xscaler = MinMaxScaler()
    yscaler = MinMaxScaler()
    X_train = Xscaler.fit_transform(X_train)
    X_test = Xscaler.transform(X_test)
    y_train, y_test = y_train.values.reshape(-1,
                                             1), y_test.values.reshape(-1, 1)
    y_train = yscaler.fit_transform(y_train)
    y_test = yscaler.transform(y_test)
    return X_train, y_train, X_test, y_test


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
    scaled = scale(X_train, y_train, X_test, y_test)

    return scaled


def train(X_train, y_train, X_test, y_test, **kwargs):
    lgb = LGBMRegressor(
        **kwargs
    )

    lgb.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)
                      ], callbacks=[early_stopping(stopping_rounds=100)]
            )
    score = lgb.score(X_train, y_train)
    print("Training score :", score)
    return lgb


def plot_test(lgb, X_test, y_test):
    y_pred = lgb.predict(X_test)
    score = lgb.score(X_test, y_test)
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
        LGBMRegressor(), grid, cv=5, refit=True, verbose=2
    )
    gs_cv.fit(X_train, y_train)
    print(gs_cv.best_params_)
    best_model = gs_cv.best_estimator_
    return best_model


def main():
    # Chargement des données
    X_train, y_train, X_test, y_test = load_data(DATA_FILENAME)

    lgb = train(X_train, y_train, X_test, y_test,
                n_estimators=10000,
                learning_rate=0.01,
                num_leaves=20,
                verbosity=1
                )
    # grid = {"n_estimators": [500, 1000, 1500],
    #         "max_depth": [1, 3, 5, 10],
    #         "learning_rate": [0.01],
    #         "verbosity": [1]}
    # xgb = grid_search(X_train, y_train,
    #                   **grid
    #                   )

    plot_test(lgb, X_test, y_test)


if __name__ == "__main__":
    main()
