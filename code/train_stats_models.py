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
from functools import partial

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import smape
from plotnine import (aes, coord_flip, geom_boxplot, ggplot, ggsave, labs,
                      scale_y_continuous)
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoCES, AutoETS, SeasonalNaive

# %%
os.chdir("S:\Python\projects\exploration")


##########################
# Load data
##########################
# %%
def load_data(dataType):
    if dataType == "daily":
        trainFile = os.path.join(os.getcwd(), "data/daily-train.csv")
        testFile = os.path.join(os.getcwd(), "data/daily-test.csv")
    elif dataType == "weekly":
        trainFile = os.path.join(os.getcwd(), "data/weekly-train.csv")
        testFile = os.path.join(os.getcwd(), "data/weekly-test.csv")
    elif dataType == "monthly":
        trainFile = os.path.join(os.getcwd(), "data/monthly-train.csv")
        testFile = os.path.join(os.getcwd(), "data/monthly-test.csv")
    elif dataType == "quarterly":
        trainFile = os.path.join(os.getcwd(), "data/quarterly-train.csv")
        testFile = os.path.join(os.getcwd(), "data/quarterly-test.csv")

    train = pd.read_csv(trainFile)
    train["ds"] = pd.to_datetime(train["ds"]).dt.normalize()
    test = pd.read_csv(testFile)
    test["ds"] = pd.to_datetime(test["ds"]).dt.normalize()

    return train, test


##########################
# Train models
##########################
# %%
def predict_stats_models(train, test, dataset):
    if dataset == "daily":
        sp = 365
        freq = "D"
    elif dataset == "weekly":
        sp = 52
        freq = "W"
    elif dataset == "monthly":
        sp = 12
        freq = "M"
    elif dataset == "quarterly":
        sp = 4
        freq = "Q"

    models = [
        SeasonalNaive(season_length=sp),
        AutoARIMA(season_length=sp),
        AutoETS(season_length=sp),
        AutoCES(season_length=sp),
    ]
    # models = [SeasonalNaive(season_length=sp)]
    sf = StatsForecast(models=models, freq=freq, n_jobs=8)
    sf.fit(df=train, sort_df=True)

    predDF = sf.predict(h=10).reset_index()

    predDF = test.merge(right=predDF, on=["unique_id", "ds"], how="inner").drop(
        ["ID"], axis=1
    )

    predDF["data"] = dataset
    predDF = predDF.melt(
        id_vars=["data", "unique_id", "ds", "y"], var_name="model", value_name="y_hat"
    )
    predDF = predDF[["data", "model", "unique_id", "ds", "y", "y_hat"]]

    # save results dataset by dataset
    fn = "stats_" + dataset + ".csv"
    fn = os.path.join("progress_data", fn)
    fn = os.path.join("data", fn)
    fn = os.path.join(os.getcwd(), fn)
    predDF.to_csv(fn, index=False)

    return predDF


def wrapper(dataset):
    print("datatset: " + dataset + ".")
    train, test = load_data(dataset)
    predDF = predict_stats_models(train, test, dataset)
    return predDF


##########################
# train models
##########################

# %%
datasets = ["monthly", "quarterly"]  # Daily OOM
modelPredictions = list(map(wrapper, datasets))
modelPredictions = pd.concat(modelPredictions)


# %%
def check_rows(dataset, modelPredictions):
    modelPredictions = modelPredictions.loc[modelPredictions.data == dataset]
    _, test = load_data(dataset)
    B = test.shape[0] * 3 == modelPredictions.shape[0]
    return B


temp = partial(check_rows, modelPredictions=modelPredictions)
all(list(map(temp, datasets)))

# %%
fn = os.path.join("data", "predictionsStatsDF.csv")
fn = os.path.join(os.getcwd(), fn)
modelPredictions.to_csv(
    fn,
    index=False,
)

# %%
fn = os.path.join("data", "predictionsStatsDF.csv")
fn = os.path.join(os.getcwd(), fn)
modelPredictions = pd.read_csv(fn)


################################################
# Summarize
################################################
# %%
def calc_smape(df, Y="y", YHAT="y_hat"):
    out = smape(df[Y], df[YHAT])
    return out


metricDF = modelPredictions.groupby(
    ["data", "model", "unique_id"], as_index=False
).apply(calc_smape)
metricDF.columns.values[3] = "smape"
metricDF = metricDF.sort_values(by=["data", "model", "unique_id"])

# %%
graph = (
    ggplot(metricDF, aes(x="data", y="smape", fill="model"))
    + geom_boxplot(alpha=0.40)
    + labs(
        title="Model Performance",
        x="Data",
        y="Symmetric Mean Absolute Percentage Error",
    )
    + scale_y_continuous(breaks=np.arange(0, 2.1, 0.2))
    + coord_flip(ylim=[0, 2])
)
ggsave(
    graph,
    os.path.join(os.getcwd(), "graphs\stats_models.png"),
    width=10,
    height=10,
)
graph

# %%
