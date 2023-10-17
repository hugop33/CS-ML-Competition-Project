from xgboost import XGBRegressor
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from src.Preprocessing import *
from config import *


def train(X_train, y_train, X_test, y_test, **kwargs):
    xgb = XGBRegressor(
        **kwargs,
        random_state=42
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
    X_train, y_train, X_test, y_test = data_pipeline(DATA_FILENAME)

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
