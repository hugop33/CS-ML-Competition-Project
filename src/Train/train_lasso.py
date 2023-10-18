import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from config import *
import os

from src.Preprocessing import data_pipeline_1, data_pipeline_2


def load_data(csv_name):
    csv_path = os.path.join(DATA_FOLDER, csv_name)
    return pd.read_csv(csv_path, sep=';')


def lasso(df: pd.DataFrame):

    X_train, y_train, X_test, y_test = data_pipeline_1(DATA_FILENAME)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    prediction = lasso.predict(X_test)
    return lasso


def lasso_predict(model, X_test):
    return model.predict(X_test)


if __name__ == "__main__":
    model = lasso("a")
    X_train, y_train, X_test, y_test = data_pipeline_2(
        DATA_FILENAME, lambda x: lasso_predict(model, x))
    print("training shape", X_train.shape)
