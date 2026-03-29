import os
import json
import joblib
import pandas as pd
from xgboost import XGBRegressor


def train():
    # SageMaker environment paths
    train_dir = os.environ.get("SM_CHANNEL_TRAINING", ".")
    model_dir = os.environ.get("SM_MODEL_DIR", "model")

    print("Loading clustered FD003 training data for RUL regression...")
    train_path = os.path.join(train_dir, "train_clustered_FD003.parquet")
    df = pd.read_parquet(train_path)

    print("Training dataset shape:", df.shape)
    print("Max RUL before clipping:", df["RUL"].max())
    print(df["RUL"].describe())

    # Piecewise linear RUL target
    # Standard CMAPSS practice to stabilize regression
    MAX_RUL = 125
    df["RUL"] = df["RUL"].clip(upper=MAX_RUL)

    print("Max RUL after clipping:", df["RUL"].max())
    print(df["RUL"].describe())

    target_col = "RUL"

    # Drop leakage / identifier / non-feature columns
    drop_cols = [
        "engine_id",
        "cycle",
        "RUL",
        "label_30",
        "cluster_name",
        "cycle_norm"
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    feature_columns = X.columns.tolist()

    print(f"Training XGBRegressor on {df.shape[0]} rows with {len(feature_columns)} features...")
    print("Feature columns:")
    print(feature_columns)

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1
    )

    model.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "xgboost_rul_model_FD003.joblib")
    features_path = os.path.join(model_dir, "rul_feature_columns.json")

    joblib.dump(model, model_path)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=4)

    print(f"Successfully saved model to: {model_path}")
    print(f"Successfully saved features to: {features_path}")


if __name__ == "__main__":
    train()