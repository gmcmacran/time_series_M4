################################################
# Overview
#
# Simple script to load predictions and calculate
# metrics. These metrics are saved to csvs.
################################################

# %%
import os
from functools import reduce

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import smape

# %%
os.chdir("S:\Python\projects\exploration")


# %%
################################################
# Define functions
################################################
def load_predictions(fn):
    fn = os.path.join("data", fn)
    fn = os.path.join(os.getcwd(), fn)
    modelPredictions = pd.read_csv(fn)
    return modelPredictions


def calc_smape(df, Y="y", YHAT="y_hat"):
    out = smape(df[Y], df[YHAT])
    return out


# %%
################################################
# Load
################################################
fns = [
    "predictionsBaselineDF.csv",
    "predictionsMlDF.csv",
    "predictionsDeepLearningDF.csv",
]
modelPredictions = map(load_predictions, fns)
modelPredictions = reduce(lambda x, y: pd.concat([x, y]), modelPredictions)
modelPredictions.model.nunique() == 7

# %%
modelPredictions.dropna().shape[0] == modelPredictions.shape[0]

# %%
np.all(np.isfinite(modelPredictions.y_hat))

# %%
################################################
# Summarize
################################################
metric_df = modelPredictions.groupby(
    ["data", "model", "unique_id"], as_index=False
).apply(calc_smape)
metric_df.columns.values[3] = "smape"
metric_df = metric_df.sort_values(by=["data", "model", "unique_id"])

metric_df["data"] = metric_df["data"].astype("category")
metric_df["data"] = metric_df["data"].cat.reorder_categories(
    ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
)

# %%
fn = os.path.join("data", "metrics_df.csv")
fn = os.path.join(os.getcwd(), fn)
metric_df.to_csv(
    fn,
    index=False,
)
