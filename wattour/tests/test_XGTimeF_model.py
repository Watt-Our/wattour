import pandas as pd

from wattour.forecasting.internal import XGBTimeFeaturesRegressor

input_df = pd.DataFrame({"timestamp": pd.date_range(start="2023-01-01", periods=24, freq="H")})


def main():
    model = XGBTimeFeaturesRegressor(num_folds=3)
    model.load(f"wattour/forecasting/internal/xgboost_models/test_pjm/model_{i}.ubj" for i in range(3))
    predictions = model.predict(input_df)
    print(predictions)


if __name__ == "__main__":
    main()
