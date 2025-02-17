import pandas as pd

from wattour.core.lmp import LMP
from wattour.forecasting.internal import XGBTimeFeaturesRegressor

input_df = pd.DataFrame({"timestamp": pd.date_range(start="2023-01-01", periods=24, freq="H")})


def main():
    model = XGBTimeFeaturesRegressor(num_folds=3)
    model.load(f"wattour/forecasting/internal/xgboost_models/test_pjm/model_{i}.ubj" for i in range(3))
    print("Model loaded.")
    predictions = model.predict_to_df(input_df)
    print("Predictions:")
    print(predictions)
    head = LMP(price = 31.5, timestamp=input_df["timestamp"].iloc[0] - pd.Timedelta(hours=1))
    timeseries_predictions = model.predict(head, input_df)
    print(timeseries_predictions)
    timeseries_predictions.plot()


if __name__ == "__main__":
    main()
