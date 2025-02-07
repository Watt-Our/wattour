import datetime

import pandas as pd

from wattour.core import LMP, LMPTimeseriesBase
from wattour.core.battery import GenericBattery
from wattour.optimization import optimize_battery_control

# Create a battery object
battery = GenericBattery(
    usable_capacity=100,
    charge_rate=10,
    discharge_rate=10,
    charge_efficiency=0.9,
    discharge_efficiency=0.9,
    self_discharge_rate=0.01,
)

# Create a hourly sample LMP data
lmps = pd.DataFrame(
    {
        "timestamp": pd.date_range("2021-01-01", periods=11, freq="h"),
        "lmp": [0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 100],
    }
)

# Create 5-minute sample LMP data
lmps_5min = pd.DataFrame(
    {
        "timestamp": pd.date_range("2021-01-01", periods=24 * 12, freq="5min"),
        "lmp": [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
            200,
            210,
            220,
            230,
            240,
        ]
        * 12,
    }
)

# Create a LMPTimeseries object
lmp_timeseries = LMPTimeseriesBase(LMP(0, lmps.iloc[0]["timestamp"], lmps.iloc[0]["lmp"]))
last_node = lmp_timeseries.create_branch_from_df(lmp_timeseries.head, lmps.iloc[1:])
lmp_timeseries.add_dummy_node(last_node, datetime.timedelta(minutes=15))
lmp_timeseries.calc_coefficients()

# Create a LMPTimeseries object with 5-minute data
lmp_timeseries_5min = LMPTimeseriesBase(LMP(0, lmps_5min.iloc[0]["timestamp"], lmps_5min.iloc[0]["lmp"]))
last_node = lmp_timeseries_5min.create_branch_from_df(lmp_timeseries_5min.head, lmps_5min.iloc[1:])
lmp_timeseries_5min.add_dummy_node(last_node, datetime.timedelta(minutes=15))
lmp_timeseries_5min.calc_coefficients()


if __name__ == "__main__":
    results = optimize_battery_control(battery, lmp_timeseries)

    results_5min = optimize_battery_control(battery, lmp_timeseries_5min)
    print(results_5min.lmp_timeseries.head)  # need to get charge differently
