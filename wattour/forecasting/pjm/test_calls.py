import pandas as pd

from wattour.core import LMPTimeseriesBase
from wattour.core.lmp_timeseries_base import transform
from wattour.forecasting.pjm import get_latest_price, get_node_fivemin

PJM_MAP = {"total_lmp_rt": "price", "datetime_beginning_utc": "timestamp"}


# test for now
if __name__ == "__main__":
    print(get_latest_price("32412297"))
    df = get_node_fivemin("32412297")
    df["datetime_beginning_utc"] = pd.to_datetime(df["datetime_beginning_utc"])
    df = transform(df, PJM_MAP)
    ts = LMPTimeseriesBase().create_branch_from_df(df)
    for node in ts.iter_nodes():
        print(node)
