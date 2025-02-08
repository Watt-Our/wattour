from wattour.forecasting.internal.xgboost.regressor_base import XGBRegressorBase


class XGBTimeFeaturesRegressor(XGBRegressorBase):
    def create_features(self, _df):
        return super().create_features(_df)
