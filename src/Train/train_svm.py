import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from config import *

from src.Preprocessing import *


def support_vector(X_train, y_train, X_test, y_test, **kwargs):
    """
    Support vector regression model

    Args:
    -----
        `X_train` (pd.DataFrame): training data
        `y_train` (pd.DataFrame): training labels
        `X_test` (pd.DataFrame): testing data
        `y_test` (pd.DataFrame): testing labels
        `**kwargs`: arguments to pass to the SVR model

    Returns:
    --------
        `ls` (SVR): SVR model
    """
    ls = SVR()
    ls.fit(X_train, y_train)
    return ls


def grid_search(X_train, y_train, **grid):
    """
    Performs a grid search on the SVR model

    Args:
    -----
        `X_train` (pd.DataFrame): training data
        `y_train` (pd.DataFrame): training labels
        `**grid`: grid of parameters to test

    Returns:
    --------
        `best_model` (SVR): best SVR model
    """
    gs_cv = GridSearchCV(
        SVR(), grid
    )
    gs_cv.fit(X_train, y_train)
    print(gs_cv.best_params_)
    best_model = gs_cv.best_estimator_
    return best_model


def plot_test(model, X_test, y_test):
    """
    Plots the predictions of the model on the test set

    Args:
    -----
        `model` (SVR): SVR model
        `X_test` (pd.DataFrame): testing data
        `y_test` (pd.DataFrame): testing labels
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE :", mse)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("SVR")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_pipeline_1(DATA_FILENAME)
    # model = support_vector(X_train, y_train, X_test, y_test)

    grid = {"kernel": ['linear', 'poly', 'rbf'],
            "C": [0.01, 0.1, 1, 10, 100],
            }
    if grid['kernel'] == 'poly':
        grid['degree'] = [2, 3, 4]

    grd = grid_search(X_train, y_train,
                      **grid
                      )

    plot_test(grd, X_test, y_test)
