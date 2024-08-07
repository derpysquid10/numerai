import numpy as np
import pandas as pd
import scipy
from pathlib import Path
import json
from scipy.stats import skew
import numerapi

ERA_COL = "era"
TARGET_COL = "target"
DATA_TYPE_COL = "data_type"
DATA_VERSION = "v4.3"


def read_prediction_data(features, training_data_path= "v4.3/train_int8.parquet", 
                         validation_data_path = "v4.3/validation_int8.parquet"
                       , live_feature_path = "v4.3/live_int8.parquet"):
    print('Reading prediction data...')

    # read in just those features along with era and target columns
 
    training_data = pd.read_parquet(training_data_path, 
                                    columns=["era", "target"] + features
                                    )

    validation_data = pd.read_parquet("4.3/validation_int8.parquet",
                                      columns=["era", "data_type", "target"] + features
                                      )
    
    live_features = pd.read_parquet(live_feature_path,
                                    columns=features
                                    )
    print("Done!")

    return training_data, validation_data, live_features