from xgboost import XGBRegressor
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from src.Preprocessing import *
from config import *


def train1(X_train, y_train, X_test, y_test, **kwargs):
    """
    Trains a XGBoost model

    Args:
    -----
        `X_train` (pd.DataFrame): training data
        `y_train` (pd.DataFrame): training labels
        `X_test` (pd.DataFrame): testing data
        `y_test` (pd.DataFrame): testing labels
        `**kwargs`: arguments to pass to the XGBoost model

    Returns:
    --------
        `xgb` (XGBRegressor): trained XGBoost model
    """
    xgb = XGBRegressor(
        **kwargs,
        random_state=42
    )

    xgb.fit(X_train, y_train, early_stopping_rounds=100,
            eval_set=[(X_train, y_train), (X_test, y_test)
                      ],
            verbose=2
            )
    score = xgb.score(X_train, y_train)
    print("Training score :", score)
    return xgb


def predictor(model, X_test):
    """
    Predicts the labels of the test set
    (Only used for the second phase's data pipeline)

    Args:
    -----
        `model` (XGBRegressor): trained XGBoost model
        `X_test` (pd.DataFrame): testing data

    Returns:
    --------
        `y_pred` (np.array): predictions of the model on the test set
    """
    return model.predict(X_test)


def plot_test1(model, X_test, y_test):
    """
    Plots the predictions of the model on the test set for phase 1

    Args:
    -----
        `model` (RandomForestRegressor): trained RandomForestRegressor model
        `X_test` (pd.DataFrame): testing data
        `y_test` (pd.DataFrame): testing labels
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE :", mse)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title(f"Random Forest (MSE = {round(mse, 5)})")
    plt.legend()
    plt.show()


def grid_search(X_train, y_train, X_test, y_test, **grid):
    """
    Performs a grid search on the XGBoost model

    Args:
    -----
        `X_train` (pd.DataFrame): training data
        `y_train` (pd.DataFrame): training labels
        `X_test` (pd.DataFrame): testing data
        `y_test` (pd.DataFrame): testing labels
        `**grid`: grid of parameters to test

    Returns:
    --------
        `best_model` (XGBRegressor): best XGBoost model
    """
    gs_cv = GridSearchCV(
        XGBRegressor(random_state=42), grid, cv=3, refit=True, verbose=2
    )
    gs_cv.fit(X_train, y_train, early_stopping_rounds=100,
              eval_set=[(X_train, y_train), (X_test, y_test)
                        ],
              verbose=0)
    print(gs_cv.best_params_)
    best_model = gs_cv.best_estimator_
    return best_model


def main():
    # Chargement des données
    X_train, y_train, X_test, y_test = data_pipeline_1(
        DATA_FILENAME)

    xgb = train1(X_train, y_train, X_test, y_test,
                 n_estimators=10000,
                 max_depth=2,
                 learning_rate=0.01,
                 verbosity=2
                 )
    # grid = {"n_estimators": [500, 1000, 1500, 2000],
    #         "max_depth": [1, 3, 5, 10],
    #         "learning_rate": [0.001, 0.01, 0.05]
    #         }
    # xgb = grid_search(X_train, y_train, X_test, y_test,
    #                   **grid
    #                   )
    plot_test1(xgb, X_test, y_test)


if __name__ == "__main__":
    main()
