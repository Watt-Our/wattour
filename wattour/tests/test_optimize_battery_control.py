import wattour.optimization as wo
from wattour.battery import GenericBattery
import pandas as pd
from wattour.optimization import optimize_battery_control
from wattour.lmp_timeseries import LMPTimeseries, LMP
import datetime

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
lmp_timeseries = LMPTimeseries(LMP())
lmp_timeseries.create_from_df(lmps, datetime.timedelta(hours=1))
lmp_timeseries.calc_coefficients()


if __name__ == "__main__":
    # # Optimize battery control with hourly LMP data
    # results_hourly = wo.optimize_battery_control(battery, lmps, initial_soc=0)
    # print(results_hourly)

    # # Optimize battery control with 5-minute LMP data
    # results_fiveminute = wo.optimize_battery_control(battery, lmps_5min, initial_soc=0)
    # print(results_fiveminute)

    # Test optimize_battery_control function

    results = optimize_battery_control(battery, lmp_timeseries)
    print(results)
