import os
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier


def train():
    train_dir = os.environ.get("SM_CHANNEL_TRAINING")
    model_dir = os.environ.get("SM_MODEL_DIR")

    print("Loading clustered FD003 training data for failure classification...")

    train_path = os.path.join(train_dir, "train_clustered_FD003.parquet")
    df = pd.read_parquet(train_path)

    print("Training dataset shape:", df.shape)

    target_col = "label_30"
    drop_cols = ["engine_id", "cycle", "RUL", "label_30", "cluster_name"]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    feature_columns = X.columns.tolist()

    print(f"Training XGBClassifier with {len(feature_columns)} features...")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "xgboost_classifier_FD003.joblib")
    features_path = os.path.join(model_dir, "classifier_feature_columns.json")

    joblib.dump(model, model_path)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=4)

    print(f"Saved classifier model to: {model_path}")
    print(f"Saved classifier feature columns to: {features_path}")


if __name__ == "__main__":
    train()