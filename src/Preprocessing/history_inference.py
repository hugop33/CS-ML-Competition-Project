from config import *
from .dates import date_to_float_col
import pandas as pd


def infer_annulations(df: pd.DataFrame):
    """
    Use previous month history over years to infer the number of annulations for the current month: creates a new feature with the mean of annulations for the current month for the current row.

    Example: For a certain train line (station1 to station2)
    If: December 2018 has 40 annulations, December 2019 35, December 2020 30, December 2021 40, December 2022 35, then all rows with that (station1 to station2) line and December month will get the value of 36.
    """
    year_thresh = 2023

    months = df["mois"].unique()
    df["lines"] = df["gare_depart"]+df["gare_arrivee"]
    years = [y for y in df["annee"].unique() if y < year_thresh]
    lines = df["lines"].unique()

    for line in lines:
        for month in months:
            df.loc[(df["lines"] == line) & (df["mois"] == month),
                   "annulations_mois"] = df.loc[(df["lines"] == line) & (df["mois"] == month), "nb_annulation"].mean()

    df.drop("lines", axis=1, inplace=True)
    return df
