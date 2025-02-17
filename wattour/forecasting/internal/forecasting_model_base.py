from abc import ABC, abstractmethod
from pathlib import Path
from typing import overload

import pandas as pd

from wattour.core.lmp import LMP
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase

# from wattour.core.lmp import LMP
# from wattour.core.lmp_timeseries_base import LMPTimeseriesBase


class ForecastingModelBase(ABC):
    @abstractmethod
    def predict(head: LMP, df:pd.DataFrame) -> LMPTimeseriesBase:
        pass

    @abstractmethod
    def load(self, paths: list[Path] | Path):
        pass
