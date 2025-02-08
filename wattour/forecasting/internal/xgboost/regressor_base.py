import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import xgboost as xgb
from matplotlib import pyplot as plt
from pandera.typing import Series
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from wattour.core.lmp import LMP
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase
from wattour.forecasting.internal.forecasting_model_base import ForecastingModelBase


# base class for forecasting XGBoost regressor models
class XGBRegressorBase(ForecastingModelBase):
    class InputDataframe(pa.DataFrameModel):
        timestamp: Series[pd.Timestamp]

    def __init__(self, num_folds: int, y_col: str = "price"):
        self.num_folds = num_folds
        self.y_col = y_col
        self.regs = []

    @abstractmethod
    def create_features(self, _df: pd.DataFrame):
        """Create features for the model. This super implementation includes time features."""
        df = _df.copy()
        df = df.set_index("timestamp")
        df["day_of_week"] = df.index.dayofweek
        df["weekend"] = df.index.dayofweek >= 5
        df["minute_of_day"] = df.index.hour * 60 + df.index.minute
        df["day_of_year"] = df.index.dayofyear
        df["month"] = df.index.month

        return df[["day_of_week", "weekend", "minute_of_day", "month"]]

    def save(self, path: Path):
        output_dir = Path(path)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        for i, reg in enumerate(self.regs):
            reg.save_model(output_dir / f"model_{i}.json")

    def load(self, path: Path):
        output_dir = Path(path)
        regs = []
        for i in range(self.num_folds):
            reg = xgb.XGBRegressor()
            reg.load_model(output_dir / f"model_{i}.json")
            regs.append(reg)

        self.regs = regs
        return regs

    def validate_train_data(self, df: pd.DataFrame):
        self.InputDataframe.validate(df)
        if self.y_col not in df.columns:
            raise ValueError(f"Training data must contain y column (named '{self.y_col}')")

    def validate_test_data(self, df: pd.DataFrame):
        self.InputDataframe.validate(df)

    def train(self, df: pd.DataFrame, test_size, verbose=False, **kwargs):
        self.validate_train_data(df)
        tss = TimeSeriesSplit(n_splits=self.num_folds, test_size=test_size)
        scores = []
        pred_indxs = []
        regs = []

        if verbose:
            _, axs = plt.subplots(self.num_folds, 1, figsize=(20, 5))

        start_time = time.time()

        for i, (train_idx, test_idx) in enumerate(tss.split(df)):
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            pred_indxs.append(test_idx)

            X_train = self.create_features(df_train)
            y_train = df_train[self.y_col]
            X_test = self.create_features(df_test)
            y_test = df_test[self.y_col]

            reg = xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 500),
                objective=kwargs.get("objective", "reg:squarederror"),
                n_jobs=kwargs.get("n_jobs", -1),
            )
            reg.set_params(
                eval_metric=kwargs.get("eval_metric", "rmse"),
                early_stopping_rounds=kwargs.get("early_stopping_rounds", 100),
            )

            reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
            regs.append(reg)
            y_pred = reg.predict(X_test)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)
            if verbose:
                print(f"Training fold {i + 1}")
                axs[i].plot(X_test.index, y_pred, label="Predicted")
                axs[i].plot(X_test.index, y_test, label="Actual")
                axs[i].legend()
                _ = xgb.plot_importance(reg, height=0.9)

        end_time = time.time()
        training_time = end_time - start_time

        if verbose:
            print(f"Training Time: {training_time}")
            plt.show()
            print(f"Individual RMSE Scores: {scores}")

        self.regs = regs

        return regs, scores

    def predict(self, df: pd.DataFrame):
        if not self.regs:
            raise ValueError("The model has not been trained or loaded yet.")

        self.validate_test_data(df)
        X = self.create_features(df)
        preds = []
        for reg in self.regs:
            preds.append(reg.predict(X))


        return preds
