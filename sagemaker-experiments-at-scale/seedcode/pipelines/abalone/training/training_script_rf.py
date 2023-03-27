import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


if __name__ == "__main__":

    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--validation-file", type=str, default="validation.csv")

    args, _ = parser.parse_known_args()

    print("reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file), header=None)
    validation_df = pd.read_csv(os.path.join(args.validation, args.validation_file), header=None)

    print("building training and testing datasets")
    # We are expecting the first column to be the target
    X_train = train_df.drop(columns=0)
    X_validation = validation_df.drop(columns=0)
    y_train = train_df[0]
    y_validation = validation_df[0]

    # train
    print("training model")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, n_jobs=-1
    )

    model.fit(X_train, y_train)

    # print abs error
    print("validating model")
    train_rmse = np.sqrt(metrics.mean_squared_error(model.predict(X_train) , y_train))
    validation_rmse = np.sqrt(metrics.mean_squared_error(model.predict(X_validation) , y_validation))

    # print perf metrics
    print(f"train-rmse:{train_rmse:.5f}")
    print(f"validation-rmse:{validation_rmse:.5f}")

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)
    print(args.min_samples_leaf)