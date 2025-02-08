from abc import ABC, abstractmethod

import pandas as pd

from wattour.core.lmp_timeseries_base import LMPTimeseriesBase


class ForecastingModelBase(ABC):
    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> LMPTimeseriesBase:
        pass
