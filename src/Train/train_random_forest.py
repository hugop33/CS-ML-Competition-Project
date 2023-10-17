import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from src.Preprocessing import *
from config import *

def random_forest(X_train, y_train, X_test, y_test, **kwargs):

    n_est = 100
    rf = RandomForestRegressor(n_estimators=n_est, criterion="mse", max_samples=0.6)
    rf.fit(X_train, y_train)

    # prediction = rf.predict(X_test)
    # mse = mean_squared_error(y_test, prediction)
    # print(f"MSE, nEstimators = {n_est} : {mse}")
    return rf

def grid_search(X_train, y_train, **grid):
    gs_cv = GridSearchCV(
        RandomForestRegressor(), grid, cv=5, refit=True, verbose=2
    )
    gs_cv.fit(X_train, y_train)
    print(gs_cv.best_params_)
    best_model = gs_cv.best_estimator_
    return best_model

def plot_test(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE :", mse)

    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Random Forest")
    plt.legend()
    plt.show()

def main():
    X_train, y_train, X_test, y_test = data_pipeline(DATA_FILENAME)
    # rf = random_forest(X_train, y_train, X_test, y_test)
    

    grid = {"n_estimators": [10, 50, 100, 150, 200],
            # "max_depth" : [],
            # "min_samples_split" : [],
            # "max_features" : [],
            "max_samples": [0.4, 0.5, 0.6, 0.7]}
    grd = grid_search(X_train, y_train,
                      **grid
                      )

    plot_test(grd, X_test, y_test)

if __name__ == "__main__":
    main()