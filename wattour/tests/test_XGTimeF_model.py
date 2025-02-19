import pandas as pd

from wattour.core.lmp import LMP
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase
from wattour.forecasting.internal import XGBTimeFeaturesRegressor

input_df = pd.DataFrame({"timestamp": pd.date_range(start="2023-01-01", periods=24, freq="H", tz="UTC")})


def main():
    model = XGBTimeFeaturesRegressor(num_folds=3)
    model.load(f"wattour/forecasting/internal/xgboost_models/test_pjm/model_{i}.ubj" for i in range(3))
    print("Model loaded.")
    predictions = model.predict_to_df(input_df)
    print("Predictions:")
    print(predictions)

    tree = LMPTimeseriesBase()
    tree.append(None, LMP(price=31.5, timestamp=input_df["timestamp"].iloc[0] - pd.Timedelta(hours=1)))
    model.predict(tree, input_df)

    print(tree)
    tree.plot()
    copy = tree.copy()
    print(copy)
    copy.plot()



if __name__ == "__main__":
    main()
