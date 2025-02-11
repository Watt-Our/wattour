import pandas as pd

from wattour.core import LMPTimeseriesBase
from wattour.core.lmp_timeseries_base import transform
from wattour.forecasting.utils.pjm import get_node_fivemin

PJM_MAP = {"total_lmp_rt": "price", "datetime_beginning_utc": "timestamp"}


# test for now
if __name__ == "__main__":
    df = get_node_fivemin("510409")
    df["datetime_beginning_utc"] = pd.to_datetime(df["datetime_beginning_utc"])
    df = transform(df, PJM_MAP)
    ts = LMPTimeseriesBase.create_branch_from_df(df)
    print(ts)
