import pandas as pd

from wattour.forecasting.internal.xgboost.regressor_base import XGBRegressorBase
from wattour.forecasting.internal.xgboost.time_features_regressor import XGBTimeFeaturesRegressor


def main(regressor: XGBRegressorBase):
    data_path = input("Enter the path to the data file: ")
    test_size = int(input("Enter the test size: "))
    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    regressor.train(df, test_size, verbose=True)


if __name__ == "__main__":
    num_folds = int(input("Enter the number of folds: "))
    regressor = XGBTimeFeaturesRegressor(num_folds)
    main(regressor=regressor)

    save = input("Do you want to save the model? (y/n): ")
    if save == "y":
        path = input("Enter the path to save the model: ")
        regressor.save(path)
    else:
        print("Model not saved.")
