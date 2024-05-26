##########################################################
# Overview
#
# A script to train three deep learning models for time series
# on the exploration datasets. Use nixtlaEnv conda environment.
#
# Output:
#   A dataframe containing predictions per model per dataset.
##########################################################

# %%
import os
import time
from functools import partial

import numpy as np
import pandas as pd
from IPython.utils import io
from mlforecast import MLForecast
from sklearn.base import BaseEstimator

# %%
os.chdir("S:\Python\projects\exploration")


# %%
class Naive(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag1"]


class SeasonalNaive(BaseEstimator):
    def __init__(self, sp):
        self.sp = sp

    def fit(self, X, y):
        return self

    def predict(self, X):
        var = "lag" + str(self.sp)
        return X[var]


##########################
# Load data
##########################
# %%
def load_data(data_type):
    if data_type == "hourly":
        train_file = os.path.join(os.getcwd(), "data\\hourly-train.csv")
        test_file = os.path.join(os.getcwd(), "data\\hourly-test.csv")
    elif data_type == "daily":
        train_file = os.path.join(os.getcwd(), "data\\daily-train.csv")
        test_file = os.path.join(os.getcwd(), "data\\daily-test.csv")
    elif data_type == "weekly":
        train_file = os.path.join(os.getcwd(), "data\\weekly-train.csv")
        test_file = os.path.join(os.getcwd(), "data\\weekly-test.csv")
    elif data_type == "monthly":
        train_file = os.path.join(os.getcwd(), "data\\monthly-train.csv")
        test_file = os.path.join(os.getcwd(), "data\\monthly-test.csv")
    elif data_type == "quarterly":
        train_file = os.path.join(os.getcwd(), "data\\quarterly-train.csv")
        test_file = os.path.join(os.getcwd(), "data\\quarterly-test.csv")
    elif data_type == "yearly":
        train_file = os.path.join(os.getcwd(), "data\\yearly-train.csv")
        test_file = os.path.join(os.getcwd(), "data\\yearly-test.csv")

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    return train, test


##########################
# Train models
##########################


# %%
def predict_baseline_models(train, test, dataset):
    if dataset == "hourly":
        h = 48
        sp = 24

    elif dataset == "daily":
        h = 14
        sp = 1

    elif dataset == "weekly":
        h = 13
        sp = 1

    elif dataset == "monthly":
        h = 18
        sp = 12

    elif dataset == "quarterly":
        h = 8
        sp = 4

    elif dataset == "yearly":
        h = 6
        sp = 1
    freq = 1

    models = [Naive(), SeasonalNaive(sp=sp)]
    mlf = MLForecast(models=models, freq=freq, lags=[1, sp])

    mlf.fit(df=train)

    predDF = mlf.predict(h=h).reset_index(drop=True)
    predDF = test.merge(right=predDF, on=["unique_id", "ds"], how="inner")

    predDF["data"] = dataset
    predDF = predDF.melt(
        id_vars=["data", "unique_id", "ds", "y"], var_name="model", value_name="y_hat"
    )
    predDF = predDF[["data", "model", "unique_id", "ds", "y", "y_hat"]]

    # save results dataset by dataset
    fn = "baseline_" + dataset + ".csv"
    fn = os.path.join("progress_data", fn)
    fn = os.path.join("data", fn)
    fn = os.path.join(os.getcwd(), fn)
    predDF.to_csv(fn, index=False)

    return predDF


def wrapper(dataset):
    start = time.time()

    print("datatset: " + dataset + ".")
    train, test = load_data(dataset)
    with io.capture_output() as captured:
        predDF = predict_baseline_models(train, test, dataset)

    end = time.time()

    timeDF = pd.DataFrame(
        data={"dataset": [dataset], "category": "Baseline", "run_time": [end - start]}
    )
    fn = os.path.join("data", f"baseline_compute_time_{dataset}.csv")
    fn = os.path.join(os.getcwd(), fn)
    timeDF.to_csv(fn, index=False)

    return predDF


##########################
# train baseline models
##########################

# %%
datasets = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
modelPredictions = list(map(wrapper, datasets))
modelPredictions = pd.concat(modelPredictions)


# %%
def check_rows(dataset, modelPredictions):
    modelPredictions = modelPredictions.loc[modelPredictions.data == dataset]
    _, test = load_data(dataset)
    B = test.shape[0] * 2 == modelPredictions.shape[0]
    return B


temp = partial(check_rows, modelPredictions=modelPredictions)
all(list(map(temp, datasets)))

# %%
fn = os.path.join("data", "predictionsBaselineDF.csv")
fn = os.path.join(os.getcwd(), fn)
modelPredictions.to_csv(
    fn,
    index=False,
)

# %%
