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

############################################
model_name = "test"
start_time = time.time()
############################################

# read the metadata and display
feature_metadata = json.load(open(f"data/features.json"))


feature_sets = feature_metadata["feature_sets"]

# Define our feature set
features = feature_sets["medium"]


# read the training and validation data given the predefined features stored in parquets as pandas DataFrames
training_data, validation_data = read_learning_data(features)
# extract feature matrix and target vector used for training
X_train = training_data.filter(like='feature_', axis='columns')
y_train = training_data[TARGET_COL]
# extract feature matrix and target vector used for validation
X_val = validation_data.filter(like='feature_', axis='columns')
y_val = validation_data[TARGET_COL]
# "garbage collection" (gc) gets rid of unused data and frees up memory
gc.collect()


num_feature_neutralization = 40  # parameter for feature neutralization used after we make our predictions
params ={"n_estimators": 500,
          "learning_rate": 0.0003,
          "task_type":"GPU",
          "depth": 12,
          "l2_leaf_reg": 21}

model = CatBoostRegressor(**params)


#spinner.start('Training model')
print("training model")
model.fit(X_train, y_train, early_stopping_rounds = 10, verbose = True)
#spinner.succeed()
print("done training!!")


spinner.start('Saving model')
model.save_model(f'models/{model_name}.txt')
spinner.succeed()



# Load the validation data and filter for data_type == "validation"
validation = pd.read_parquet(
    f"data/validation_int8.parquet",
    columns=["era", "data_type", "target"] + features
)
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]

# Downsample to every 4th era to reduce memory usage and speedup evaluation (suggested for Colab free tier)
# Comment out the line below to use all the data (slower and higher memory usage, but more accurate evaluation)
validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

# Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
# so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
#last_train_era = int(train["era"].unique()[-1])
#eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
#validation = validation[~validation["era"].isin(eras_to_embargo)]

validation.loc[:, f"preds_{model_name}"] = model.predict(validation[features])


#validation["prediction"] = model.predict(validation[features])

spinner.start('Neutralizing')
neutralize_riskiest_features(training_data, validation, features, model_name, k=num_feature_neutralization)
spinner.succeed()


validation["prediction"] = validation[f"preds_{model_name}_neutral_riskiest_{num_feature_neutralization}"] \
    .rank(pct=True)
validation["prediction"].to_csv(f"predictions/{model_name}.csv")


# Validation Stage
###########################################################################################

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

##############################################################################
end_time = time.time()
time_elapsed_mins = (end_time - start_time)//60
time_elapsed_secs = (end_time - start_time)%60
print(f"Time elapsed: {time_elapsed_mins} mins {time_elapsed_secs} secs")

print('Done!')
##############################################################################
