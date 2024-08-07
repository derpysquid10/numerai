from numerapi import NumerAPI
import pandas as pd

napi = NumerAPI()
# Use int8 to save on storage and memory
napi.download_dataset("v4.3/train_int8.parquet")
training_data = pd.read_parquet("v4.3/train_int8.parquet")

from catboost import CatBoostRegressor


features = [f for f in training_data.columns if "feature" in f]
model = CatBoostRegressor(
      n_estimators=50,
      learning_rate=0.0003,
        task_type = "GPU",
        l2_leaf_reg = 9,

      max_depth=13
)
model.fit(
      training_data[features],
      training_data["target"]
)

# Authenticate
napi = NumerAPI("N2EXYEACXRCKGUGGA6F3ESBH7BCGQMAZ", "B5ROW2MKVRRTTTQFNWKJ37MCP6EFDUULCH5ST7VRVTHBFWLVVKNTKQEKLZDOIZGD")

# Get current round
current_round = napi.get_current_round()

# Download latest live features
napi.download_dataset(f"v4.1/live_{current_round}.parquet")
live_data = pd.read_parquet(f"v4.1/live_{current_round}.parquet")
live_features = live_data[[f for f in live_data.columns if "feature" in f]]

# Generate live predictions
live_predictions = model.predict(live_features)

# Format submission
submission = pd.Series(live_predictions, index=live_features.index).to_frame("prediction")
submission.to_csv(f"prediction_{current_round}.csv")

# Upload submission 
napi.upload_predictions(f"prediction_{current_round}.csv", model_id="your-model-id")