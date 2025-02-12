from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from wattour.core.lmp import LMP
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase


class ForecastingModelBase(ABC):
    @abstractmethod
    def load(self, paths: list[Path] | Path):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
