import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from src.Preprocessing import *
from config import *


def random_forest1(X_train, y_train, **kwargs):
    """
    Trains a RandomForestRegressor model

    Args:
    -----
        `X_train` (pd.DataFrame): training data
        `y_train` (pd.DataFrame): training labels
        `X_test` (pd.DataFrame): testing data
        `y_test` (pd.DataFrame): testing labels
        `**kwargs`: arguments to pass to the RandomForestRegressor model

    Returns:
    --------
        `rf` (RandomForestRegressor): trained RandomForestRegressor model
    """
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)

    return rf


def grid_search(X_train, y_train, **grid):
    """
    Performs a grid search on the RandomForestRegressor model

    Args:
    -----
        `X_train` (pd.DataFrame): training data
        `y_train` (pd.DataFrame): training labels
        `**grid`: grid of parameters to test

    Returns:
    --------
        `best_model` (RandomForestRegressor): best RandomForestRegressor model
    """
    gs_cv = GridSearchCV(
        RandomForestRegressor(), grid, cv=5, refit=True, verbose=2
    )
    gs_cv.fit(X_train, y_train)
    print(gs_cv.best_params_)
    best_model = gs_cv.best_estimator_
    return best_model


def plot_test(model, X_test, y_test):
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


def main():
    X_train, y_train, X_test, y_test = data_pipeline_1(DATA_FILENAME)
    rf = random_forest1(X_train, y_train)
    plot_test(rf, X_test, y_test)

    # grid = {"n_estimators": [10, 50, 100, 150, 200],
    #         # "max_depth" : [],
    #         # "min_samples_split" : [],
    #         # "max_features" : [],
    #         "max_samples": [0.4, 0.5, 0.6, 0.7]}
    # grd = grid_search(X_train, y_train,
    #                   **grid
    #                   )

if __name__ == "__main__":
    main()
