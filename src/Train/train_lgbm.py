from lightgbm import LGBMRegressor, early_stopping
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.Preprocessing import *
from config import *


def train(X_train, y_train, X_test, y_test, **kwargs):
    lgb = LGBMRegressor(
        **kwargs,
        random_state=42
    )

    lgb.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)
                      ], callbacks=[early_stopping(stopping_rounds=100)]
            )
    score = lgb.score(X_train, y_train)
    print("Training score :", score)
    return lgb


def lgbm_predict(model, X_test):
    return model.predict(X_test)


def plot_test(lgb, X_test, y_test):
    cols = y_test.columns
    X_test, y_test = X_test.values, y_test.values
    nb_cols = y_test.shape[1]
    y_pred = lgb.predict(X_test)
    score = lgb.score(X_test, y_test)
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
    gs_cv = GridSearchCV(
        LGBMRegressor(), grid, cv=5, refit=True, verbose=2
    )
    gs_cv.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)
                                          ], callbacks=[early_stopping(stopping_rounds=100)])
    print(gs_cv.best_params_)
    best_model = gs_cv.best_estimator_
    return best_model


def main():
    # Chargement des données
    X_train, y_train, X_test, y_test = data_pipeline_1(DATA_FILENAME)

    lgb = train(X_train, y_train, X_test, y_test,
                n_estimators=10000,
                learning_rate=0.01,
                max_depth=2,
                verbosity=1
                )
    # grid = {"n_estimators": [1000, 1500, 2000],
    #         "max_depth": [1, 3, 5, 10],
    #         "learning_rate": [0.001, 0.01, 0.05],
    #         "verbosity": [0]}
    # lgb = grid_search(X_train, y_train, X_test, y_test,
    #                   **grid
    #                   )
    X_train, y_train, X_test, y_test = data_pipeline_2(
        DATA_FILENAME, lambda x: lgb.predict(x))
    lgb2 = train(X_train, y_train, X_test, y_test,
                 n_estimators=10000,
                 max_depth=2,
                 learning_rate=0.01,
                 verbosity=2
                 )

    plot_test(lgb2, X_test, y_test)


if __name__ == "__main__":
    main()
