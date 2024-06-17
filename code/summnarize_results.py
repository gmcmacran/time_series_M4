# %%
import os
from functools import reduce

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import smape
from plotnine import (aes, coord_flip, geom_boxplot, geom_col, geom_hline,
                      ggplot, ggsave, labs, scale_y_continuous)

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


def load_compute_time(category):
    def load_dataset(dataset):
        fn = f"{category}_compute_time_{dataset}.csv"
        fn = os.path.join("data", fn)
        fn = os.path.join(os.getcwd(), fn)
        compute_df = pd.read_csv(fn)

        return compute_df

    dfs = map(
        load_dataset, ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
    )
    compute_df = reduce(lambda x, y: pd.concat([x, y]), dfs)
    return compute_df


def calc_smape(df, Y="y", YHAT="y_hat"):
    out = smape(df[Y], df[YHAT])
    return out


################################################
# Summarize
################################################

# %%
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
graph = (
    ggplot(metric_df, aes(x="data", y="smape", fill="model"))
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
    os.path.join(os.getcwd(), "graphs\\all_models.png"),
    width=10,
    height=10,
)
graph

# %%
temp = metric_df.groupby("model", as_index=False)["smape"].mean()
graph = (
    ggplot(temp, aes(x="model", y="smape"))
    + geom_col(alpha=0.40)
    + labs(title="Model Performance", x="Model", y="Average Smape")
    + geom_hline(yintercept=0.11374)
    + scale_y_continuous(breaks=np.arange(0, 0.16, 0.02))
    + coord_flip()
)
ggsave(
    graph,
    os.path.join(os.getcwd(), "graphs\\all_models_avg.png"),
    width=10,
    height=10,
)
graph

# %%
temp


# %%
categories = ["baseline", "ml", "deep_learning"]
computeDF = map(load_compute_time, categories)
computeDF = reduce(lambda x, y: pd.concat([x, y]), computeDF)

computeDF["run_time"] = computeDF.run_time / (60)
computeDF["run_time"] = computeDF.run_time / (60)

computeDF["dataset"] = computeDF["dataset"].astype("category")
computeDF["dataset"] = computeDF["dataset"].cat.reorder_categories(
    ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
)

computeDF.head()

# %%
graph = (
    ggplot(computeDF, aes(x="dataset", y="run_time", fill="category"))
    + geom_col(alpha=0.40, position="dodge")
    + scale_y_continuous(breaks=np.arange(0, 12, 2))
    + labs(
        x="Data",
        y="Compute Time (Hours)",
    )
)
ggsave(
    graph,
    os.path.join(os.getcwd(), "graphs\\compute_time.png"),
    width=10,
    height=10,
)
graph


# %%
