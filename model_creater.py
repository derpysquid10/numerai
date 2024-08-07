import numpy as np
import pandas as pd
import scipy
from halo import Halo
from pathlib import Path
import json
from scipy.stats import skew
import numerapi
import gc
from numerapi import NumerAPI
import time


from catboost import CatBoostRegressor
import catboost


from utils_plus import *

import json

from numerai_tools.scoring import numerai_corr, correlation_contribution

####################################################
model_name = "test"
start_time = time.time()
####################################################


# read the metadata and display
feature_metadata = json.load(open(f"data/features.json"))

# Initialize NumerAPI - the official Python API client for Numerai
from numerapi import NumerAPI
napi = NumerAPI()

# list the datasets and available versions
all_datasets = napi.list_datasets()
dataset_versions = list(set(d.split('/')[0] for d in all_datasets))

# Set data version to one of the latest datasets
DATA_VERSION = "v4.3"

# Print all files available for download for our version
current_version_files = [f for f in all_datasets if f.startswith(DATA_VERSION)]

feature_sets = feature_metadata["feature_sets"]

# Define our feature set
feature_set = feature_sets["medium"]

# Load only the "medium" feature set to
# Use the "all" feature set to use all features
train = pd.read_parquet(
    f"{DATA_VERSION}/train_int8.parquet",
    columns=["era", "target"] + feature_set
)

# Downsample to every 4th era to reduce memory usage and speedup model training (suggested for Colab free tier)
# Comment out the line below to use all the data
train = train[train["era"].isin(train["era"].unique()[::4])]

num_feature_neutralization = 40  # parameter for feature neutralization used after we make our predictions
params ={"n_estimators": 50,
          "learning_rate": 0.00003,
          "task_type":"GPU",
          "depth": 11,
          'l2_leaf_reg': 17}

model = CatBoostRegressor(**params)

print("training model")

model.fit(
  train[feature_set],
  train["target"],
  verbose = False
)

print("training done!!!!!!!!!")

# Load the validation data and filter for data_type == "validation"
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation_int8.parquet",
    columns=["era", "data_type", "target"] + feature_set
)
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]

# Downsample to every 4th era to reduce memory usage and speedup evaluation (suggested for Colab free tier)
# Comment out the line below to use all the data (slower and higher memory usage, but more accurate evaluation)
validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

# Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
# so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

# Generate predictions against the out-of-sample validation features
# This will take a few minutes 🍵
validation["prediction"] = model.predict(validation[feature_set])
validation[["era", "prediction", "target"]]

# Load live features
live_features = pd.read_parquet(f"{DATA_VERSION}/live_int8.parquet", columns=feature_set)

# Generate live predictions
live_predictions = model.predict(live_features[feature_set])

# Format submission
pd.Series(live_predictions, index=live_features.index).to_frame("prediction")
#live_predictions["prediction"].to_csv(f"predictions/{model_name}.csv")

# Define your prediction pipeline as a function
def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    live_predictions = model.predict(live_features[feature_set])
    submission = pd.Series(live_predictions, index=live_features.index)
    return submission.to_frame("prediction")




######################################################################
# Compute the per-era corr between our predictions and the target values
per_era_corr = validation.groupby("era").apply(
    lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
)

per_era_corr.plot(
  title="Validation CORR",
  kind="bar",
  figsize=(8, 4),
  xticks=[],
  legend=False,
  snap=False
)


# Plot the cumulative per-era correlation
per_era_corr.cumsum().plot(
  title="Cumulative Validation CORR",
  kind="line",
  figsize=(8, 4),
  legend=False
)

# Compute performance metrics
corr_mean = per_era_corr.mean()
corr_std = per_era_corr.std(ddof=0)
corr_sharpe = corr_mean / corr_std
corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()
print("corr_mean: ", corr_mean)
print("corr_std: ", corr_std)
print("corr_sharpe: ", corr_sharpe)
print("corr_max_drawdown: ", corr_max_drawdown)


######################################################################

# Use the cloudpickle library to serialize your function
import cloudpickle
p = cloudpickle.dumps(predict)
with open(f'{model_name}.pkl', "wb") as f:
    f.write(p)

######################################################################

    end_time = time.time()
time_elapsed_mins = (end_time - start_time)//60
time_elapsed_secs = (end_time - start_time)%60
print(f"Time elapsed: {time_elapsed_mins} mins {time_elapsed_secs} secs")

print('Done!')