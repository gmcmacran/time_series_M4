##########################################################
# Overview
#
# A script to take raw exploration data and make it ready for
# nixtla's neuralforecast.
#
#
# Output:
# Two cleaned csvs per time unit in data folder.
##########################################################

# %%
import os

import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4


# %%
def prep_data(dataset):
    print(dataset)
    if dataset == "hourly":
        train, _, _ = M4.load(
            directory=os.path.join("S:\Python\data\exploration"), group="Hourly"
        )
        correct_series_count = 414
        h = 48

    elif dataset == "daily":
        train, _, _ = M4.load(
            directory=os.path.join("S:\Python\data\exploration"), group="Daily"
        )
        correct_series_count = 4227
        h = 14

    elif dataset == "weekly":
        train, _, _ = M4.load(
            directory=os.path.join("S:\Python\data\exploration"), group="Weekly"
        )
        correct_series_count = 359
        h = 13

    elif dataset == "monthly":
        train, _, _ = M4.load(
            directory=os.path.join("S:\Python\data\exploration"), group="Monthly"
        )
        correct_series_count = 48000
        h = 18

    elif dataset == "quarterly":
        train, _, _ = M4.load(
            directory=os.path.join("S:\Python\data\exploration"), group="Quarterly"
        )
        correct_series_count = 24000
        h = 8

    elif dataset == "yearly":
        train, _, _ = M4.load(
            directory=os.path.join("S:\Python\data\exploration"), group="Yearly"
        )
        correct_series_count = 23000
        h = 6

    else:
        print("Error: Incorrect value of dataType.")

    test = train.sort_values(by="ds").groupby("unique_id").tail(h)
    train = train.loc[~train.index.isin(test.index)]

    # confirm there is the right amount of series.
    if train.unique_id.nunique() != correct_series_count:
        print("Error: Incorrect number of series after shaping.")
    if test.unique_id.nunique() != correct_series_count:
        print("Error: Incorrect number of series after shaping.")

    if dataset == "hourly":
        train_file = os.path.join(
            "S:\Python\projects\exploration\data", "hourly-train.csv"
        )
        test_file = os.path.join(
            "S:\Python\projects\exploration\data", "hourly-test.csv"
        )
    elif dataset == "daily":
        train_file = os.path.join(
            "S:\Python\projects\exploration\data", "daily-train.csv"
        )
        test_file = os.path.join(
            "S:\Python\projects\exploration\data", "daily-test.csv"
        )
    elif dataset == "weekly":
        train_file = os.path.join(
            "S:\Python\projects\exploration\data", "weekly-train.csv"
        )
        test_file = os.path.join(
            "S:\Python\projects\exploration\data", "weekly-test.csv"
        )
    elif dataset == "monthly":
        train_file = os.path.join(
            "S:\Python\projects\exploration\data", "monthly-train.csv"
        )
        test_file = os.path.join(
            "S:\Python\projects\exploration\data", "monthly-test.csv"
        )
    elif dataset == "quarterly":
        train_file = os.path.join(
            "S:\Python\projects\exploration\data", "quarterly-train.csv"
        )
        test_file = os.path.join(
            "S:\Python\projects\exploration\data", "quarterly-test.csv"
        )
    elif dataset == "yearly":
        train_file = os.path.join(
            "S:\Python\projects\exploration\data", "yearly-train.csv"
        )
        test_file = os.path.join(
            "S:\Python\projects\exploration\data", "yearly-test.csv"
        )

    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)


# %%
datasets = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
list(map(prep_data, datasets))

# %%
