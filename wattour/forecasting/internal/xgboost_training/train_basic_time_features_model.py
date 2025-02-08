import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def create_features(_df: pd.DataFrame):
    df = _df.copy()
    df = df.set_index("time")
    df.index = pd.to_datetime(df.index)
    df["day_of_week"] = df.index.dayofweek
    df["weekend"] = df.index.dayofweek >= 5
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month

    return df[["day_of_week", "weekend", "minute_of_day", "month"]], df[
        "price"
    ]


def train_basic_time_features_model(df: pd.DataFrame, num_folds, test_size):
    tss = TimeSeriesSplit(n_splits=num_folds, test_size=test_size)

    preds = []
    scores = []
    pred_indxs = []
    regs = []
    _, axs = plt.subplots(num_folds, 1, figsize=(20, 5))

    for i, (train_idx, test_idx) in enumerate(tss.split(df)):
        print(f"Training fold {i + 1}")
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        pred_indxs.append(test_idx)

        X_train, y_train = create_features(df_train)
        X_test, y_test = create_features(df_test)

        reg = xgb.XGBRegressor(n_estimators=500, objective="reg:squarederror", n_jobs=-1)
        reg.set_params(eval_metric="rmse", early_stopping_rounds=100)
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
        regs.append(reg)
        y_pred = reg.predict(X_test)
        preds.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)

        axs[i].plot(X_test.index, y_pred, label="Predicted")
        axs[i].plot(X_test.index, y_test, label="Actual")
        axs[i].legend()
        _ = xgb.plot_importance(reg, height=0.9)

        i += 1

    print(f"Individual RMSE Scores: {scores}")

    return preds, scores, pred_indxs, regs


if __name__ == "__main__":
    data_path = sys.argv[1]
    num_folds = int(sys.argv[2])
    test_size = int(sys.argv[3])
    data_output = sys.argv[4]

    df = pd.read_csv(data_path)
    preds, scores, pred_indxs, regs = train_basic_time_features_model(df, num_folds, test_size)

    script_dir = Path("/Users/alextseng/MacDocuments/GitHub/wattour/wattour/forecasting/internal/xgboost_models")
    output_dir = script_dir / data_output

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for i, reg in enumerate(regs):
        reg.save_model(output_dir / f"model_{i}.json")

    plt.show()
