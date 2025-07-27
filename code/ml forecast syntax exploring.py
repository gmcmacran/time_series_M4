##########################################################
# Overview
#
# A script explore the transformations used by MLForecast.
# Use nixtlaEnv conda environment.
#
#
# Output:
# No results are saved. Just exploration.
##########################################################

# %%
import os

import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences, LocalBoxCox, LocalStandardScaler
from mlforecast.utils import generate_daily_series
from sklearn.linear_model import Ridge

# %%
os.chdir("S:\Python\projects\exploration")


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


train, test = load_data("quarterly")
sp = 4

# %%
fcst = MLForecast(
    models=[Ridge(fit_intercept=True, alpha=3.7091571393796694)],
    freq=1,
    lags=list(range(sp, sp + 3)),
    lag_transforms={sp: [RollingMean(window_size=sp, min_samples=1)]},
    target_transforms=[LocalBoxCox(), Differences([sp]), LocalStandardScaler()],
    num_threads=8,
    dropna=False,
)
temp = fcst.preprocess(train)
temp.head(5)

# %%
temp.shape[0] == temp.dropna().shape[0]


# %%
fcst.fit(df=train)
# %%
