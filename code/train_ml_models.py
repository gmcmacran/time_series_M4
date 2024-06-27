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
import optuna
import pandas as pd
from IPython.utils import io
from mlforecast.auto import AutoLightGBM, AutoMLForecast, AutoRidge
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences, LocalBoxCox, LocalStandardScaler

# %%
os.chdir("S:\Python\projects\exploration")


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
def predict_ml_models(train, test, dataset):
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

    def lightgbm_space(trial: optuna.Trial):
        return {
            "bagging_freq": 1,
            "learning_rate": 0.05,
            "verbosity": -1,
            "objective": "mape",
            "n_estimators": trial.suggest_int("n_estimators", 20, 1000, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        }

    def init_config(trial, sp):
        if sp == 1:
            candidate_lags = [[1], [1, 2], [1, 2, 3]]
            candidate_lag_tfms = [
                {1: [RollingMean(window_size=sp)]},
                {1: [RollingMean(window_size=3)]},
                {
                    1: [ExpandingMean(), RollingMean(window_size=sp, min_samples=1)],
                },
                {
                    1: [ExpandingMean(), RollingMean(window_size=3, min_samples=1)],
                },
            ]
            candidate_targ_tfms = [
                [LocalBoxCox()],
                [Differences([1])],
                [LocalStandardScaler()],
                [LocalBoxCox(), LocalStandardScaler()],
                [LocalBoxCox(), Differences([1])],
                [LocalBoxCox(), Differences([1]), LocalStandardScaler()],
            ]
        elif sp == 4:
            candidate_lags = [
                [1],
                [1, 2, 3],
                list(range(sp, sp + 1)),
                list(range(sp, sp + 3)),
                [1, sp],
                [1, 2, 3] + list(range(sp, sp + 3)),
            ]

            candidate_lag_tfms = [
                {1: [RollingMean(window_size=sp)]},
                {
                    1: [RollingMean(window_size=sp, min_samples=1)],
                    sp: [RollingMean(window_size=sp, min_samples=1)],
                },
                {
                    sp: [RollingMean(window_size=sp, min_samples=1)],
                },
                {
                    1: [ExpandingMean(), RollingMean(window_size=sp, min_samples=1)],
                    sp: [ExpandingMean(), RollingMean(window_size=sp, min_samples=1)],
                },
            ]

            candidate_targ_tfms = [
                [LocalBoxCox()],
                [Differences([1])],
                [LocalStandardScaler()],
                [LocalBoxCox(), LocalStandardScaler()],
                [LocalBoxCox(), Differences([1])],
                [LocalBoxCox(), Differences([1]), LocalStandardScaler()],
            ]

        elif sp == 12:
            candidate_lags = [
                [1],
                [1, 2, 3],
                [1, 2, 3, 4, 5, 6],
            ]

            candidate_lag_tfms = [
                {1: [RollingMean(window_size=sp)]},
            ]

            candidate_targ_tfms = [
                [LocalBoxCox()],
                [Differences([1])],
                [LocalStandardScaler()],
                [LocalBoxCox(), LocalStandardScaler()],
                [LocalBoxCox(), Differences([1])],
                [LocalBoxCox(), Differences([1]), LocalStandardScaler()],
            ]

        elif sp == 24:
            candidate_lags = [
                [1],
                [1, 2, 3],
                list(range(sp, sp + 1)),
                list(range(sp, sp + 3)),
                [1, sp],
                [1, 2, 3] + list(range(sp, sp + 3)),
            ]

            candidate_lag_tfms = [
                {1: [RollingMean(window_size=sp)]},
                {
                    1: [RollingMean(window_size=sp, min_samples=1)],
                    sp: [RollingMean(window_size=sp, min_samples=1)],
                },
                {
                    sp: [RollingMean(window_size=sp, min_samples=1)],
                },
                {
                    1: [ExpandingMean(), RollingMean(window_size=sp, min_samples=1)],
                    sp: [ExpandingMean(), RollingMean(window_size=sp, min_samples=1)],
                },
            ]

            candidate_targ_tfms = [
                [LocalBoxCox()],
                [Differences([1])],
                [Differences([sp])],
                [LocalStandardScaler()],
                [LocalBoxCox(), LocalStandardScaler()],
                [LocalBoxCox(), Differences([1])],
                [LocalBoxCox(), Differences([sp])],
                [LocalBoxCox(), Differences([1]), LocalStandardScaler()],
                [LocalBoxCox(), Differences([sp]), LocalStandardScaler()],
                [
                    LocalBoxCox(),
                    Differences([sp]),
                    Differences([1]),
                    LocalStandardScaler(),
                ],
            ]

        lag_idx = trial.suggest_categorical("lag_idx", range(len(candidate_lags)))
        lag_tfms_idx = trial.suggest_categorical(
            "lag_tfms_idx", range(len(candidate_lag_tfms))
        )
        targ_tfms_idx = trial.suggest_categorical(
            "targ_tfms_idx", range(len(candidate_targ_tfms))
        )

        return {
            "lags": candidate_lags[lag_idx],
            "lag_transforms": candidate_lag_tfms[lag_tfms_idx],
            "target_transforms": candidate_targ_tfms[targ_tfms_idx],
        }

    init_config = partial(init_config, sp=sp)

    models = [AutoLightGBM(config=lightgbm_space), AutoRidge()]
    auto_mlf = AutoMLForecast(
        models=models,
        freq=1,
        season_length=sp,
        init_config=init_config,
        num_threads=8,
    )

    auto_mlf.fit(df=train, n_windows=1, h=h, num_samples=50)

    predDF = auto_mlf.predict(h=h).reset_index(drop=True)

    predDF = test.merge(right=predDF, on=["unique_id", "ds"], how="inner")

    predDF["data"] = dataset
    predDF = predDF.melt(
        id_vars=["data", "unique_id", "ds", "y"], var_name="model", value_name="y_hat"
    )
    predDF = predDF[["data", "model", "unique_id", "ds", "y", "y_hat"]]

    # save results dataset by dataset
    fn = "ml_" + dataset + ".csv"
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
        predDF = predict_ml_models(train, test, dataset)

    end = time.time()

    timeDF = pd.DataFrame(
        data={
            "dataset": [dataset],
            "category": "Machine Learning",
            "run_time": [end - start],
        }
    )
    fn = os.path.join("data", f"ml_compute_time_{dataset}.csv")
    fn = os.path.join(os.getcwd(), fn)
    timeDF.to_csv(fn, index=False)

    return predDF


##########################
# train ML models
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
fn = os.path.join("data", "predictionsMlDF.csv")
fn = os.path.join(os.getcwd(), fn)
modelPredictions.to_csv(
    fn,
    index=False,
)
