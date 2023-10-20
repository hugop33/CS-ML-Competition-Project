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


def plot_test(xgb, X_test, y_test):
    """
    Plots the predictions of the model on the test set

    Args:
    -----
        `xgb` (XGBRegressor): trained XGBoost model
        `X_test` (pd.DataFrame): testing data
        `y_test` (pd.DataFrame): testing labels
    """
    y_test = y_test if isinstance(
        y_test, pd.DataFrame) else pd.DataFrame(y_test)
    cols = y_test.columns
    X_test, y_test = X_test.values, y_test.values
    nb_cols = y_test.shape[1]
    y_pred = xgb.predict(X_test).reshape(-1, nb_cols)
    score = xgb.score(X_test, y_test)
    print("Test score :", score)

    mse_cols = []
    for i in range(nb_cols):
        mse_cols.append(mean_squared_error(y_test[:, i], y_pred[:, i]))
    print("MSEs :", mse_cols)

    x_ax = range(len(y_test))
    for i in range(nb_cols):
        plt.subplot(nb_cols, 1, i+1)
        plt.plot(x_ax, y_test[:, i], label=f"original {cols[i]}")
        plt.plot(x_ax, y_pred[:, i], label=f"predicted {cols[i]}")
        plt.title("Predicted and test instances, XGBoost")
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

    xgb = train(X_train, y_train, X_test, y_test,
                n_estimators=10000,
                max_depth=2,
                learning_rate=0.01,
                verbosity=2
                )
    plot_test(xgb, X_test, y_test)
    # grid = {"n_estimators": [500, 1000, 1500, 2000],
    #         "max_depth": [1, 3, 5, 10],
    #         "learning_rate": [0.001, 0.01, 0.05]
    #         }
    # xgb = grid_search(X_train, y_train, X_test, y_test,
    #                   **grid
    #                   )
    X_train, y_train, X_test, y_test = data_pipeline_2(
        DATA_FILENAME, lambda x: xgb.predict(x))
    xgb2 = train(X_train, y_train, X_test, y_test,
                 n_estimators=10000,
                 max_depth=2,
                 learning_rate=0.01,
                 verbosity=2
                 )
    plot_test(xgb2, X_test, y_test)


if __name__ == "__main__":
    main()
