##########################################################
# Overview
#
# A script to train three deep learning models for time series
# on the exploration datasets. Use nixtlaEnv conda environment.
#
# Use nixtlaEnvDeep for GPU support.
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
import torch
from IPython.utils import io
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNBEATSx, AutoNHITS, AutoTFT
from neuralforecast.losses.pytorch import SMAPE
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

# %%
torch.cuda.is_available()

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
def predict_neuralforecast_models(train, test, dataset):
    if dataset == "hourly":
        h = 48

    elif dataset == "daily":
        h = 14

    elif dataset == "weekly":
        h = 13

    elif dataset == "monthly":
        h = 18

    elif dataset == "quarterly":
        h = 8

    elif dataset == "yearly":
        h = 6

    num_samples = 50
    backend = "ray"
    search_alg = HyperOptSearch()

    base_config = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "input_size": tune.randint(int(h / 2), 3 * h),
        "max_steps": tune.randint(100, 1000),
        "batch_size": tune.choice([32, 64, 128]),
        "random_seed": 42,
        "scaler_type": tune.choice(
            [
                "identity",
                "standard",
                "robust",
                "minmax",
                "minmax1",
                "invariant",
                "revin",
            ]
        ),
        "start_padding_enabled": True,
    }

    config_00 = {
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "dropout": tune.uniform(0, 0.50),
        **base_config,
    }
    params_00 = {
        "h": h,
        "num_samples": num_samples,
        "loss": SMAPE(),
        "config": config_00,
        "search_alg": search_alg,
        "backend": backend,
    }

    config_01 = {
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "dropout_prob_theta": tune.uniform(0, 0.50),
        **base_config,
    }
    params_01 = {
        "h": h,
        "num_samples": num_samples,
        "loss": SMAPE(),
        "config": config_01,
        "search_alg": search_alg,
        "backend": backend,
    }

    config_02 = {
        "dropout_prob_theta": tune.uniform(0, 0.50),
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        **base_config,
    }
    params_02 = {
        "h": h,
        "num_samples": num_samples,
        "loss": SMAPE(),
        "config": config_02,
        "search_alg": search_alg,
        "backend": backend,
    }

    models = [AutoTFT(**params_00), AutoNBEATSx(**params_01), AutoNHITS(**params_02)]
    nf = NeuralForecast(models=models, freq=1)
    nf.fit(df=train)

    predDF = nf.predict().reset_index()

    predDF = test.merge(right=predDF, on=["unique_id", "ds"], how="inner")

    predDF["data"] = dataset
    predDF = predDF.melt(
        id_vars=["data", "unique_id", "ds", "y"], var_name="model", value_name="y_hat"
    )
    predDF = predDF[["data", "model", "unique_id", "ds", "y", "y_hat"]]

    # save results dataset by dataset
    fn = "deeplearning_" + dataset + ".csv"
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
        predDF = predict_neuralforecast_models(train, test, dataset)

    end = time.time()

    timeDF = pd.DataFrame(
        data={
            "dataset": [dataset],
            "category": "Deep Learning",
            "run_time": [end - start],
        }
    )
    fn = os.path.join("data", f"deep_learning_compute_time_{dataset}.csv")
    fn = os.path.join(os.getcwd(), fn)
    timeDF.to_csv(fn, index=False)

    return predDF


##########################
# train models
##########################

# %%
datasets = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
model_predictions = list(map(wrapper, datasets))
model_predictions = pd.concat(model_predictions)


# %%
def check_rows(dataset, model_predictions):
    model_predictions = model_predictions.loc[model_predictions.data == dataset]
    _, test = load_data(dataset)
    B = test.shape[0] * 3 == model_predictions.shape[0]
    return B


temp = partial(check_rows, model_predictions=model_predictions)
all(list(map(temp, datasets)))

# %%
list(map(temp, datasets))

# %%
fn = os.path.join("data", "predictionsDeepLearningDF.csv")
fn = os.path.join(os.getcwd(), fn)
model_predictions.to_csv(
    fn,
    index=False,
)

# %%