import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

from ..Preprocessing.pipeline import data_pipeline

def load_data(csv_name):
    csv_path = os.path.join(DATA_FOLDER, csv_name)
    return pd.read_csv(csv_path, sep=';')


def lasso(df: pd.DataFrame):

    X_train, y_train, X_test, y_test = data_pipeline(df)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    prediction = lasso.predict(X_test)
    return prediction