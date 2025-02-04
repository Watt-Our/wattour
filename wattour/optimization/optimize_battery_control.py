import time

import gurobipy as gp
from gurobipy import GRB

from wattour.core import BatteryBase, LMPTimeseriesBase

from .lmp_timeseries_gurobi import LMPTimeseriesGurobi


# Optimize battery control for given battery and lmps using LMPTimeseries. If the
# LMPTimeseries has branches, this function will complete stochastic optimization
def optimize_battery_control(
    battery: BatteryBase, lmps: LMPTimeseriesBase, initial_soc: float = 0, final_soc: float = 0
):
    if not isinstance(lmps, LMPTimeseriesGurobi):
        lmps = LMPTimeseriesGurobi().create_from_lmp_timeseries_base(lmps)

    # Check that initial and final state of charge are valid
    max_soc = battery.get_usable_capacity()  # maximum state of charge
    if initial_soc > max_soc or initial_soc < 0:
        raise ValueError("Invalid initial state of charge")
    if final_soc > max_soc or final_soc < 0:
        raise ValueError("Invalid final state of charge")

    model = gp.Model("Battery Control Optimizer")

    if lmps.head.coefficient is None:
        lmps.calc_coefficients()

    lmps.add_gurobi_vars(model)
    node_list = lmps.get_node_list(dummies=False)

    # Objective function; charge and dischare are in power units
    model.setObjective(
        gp.quicksum(
            (node_list[i].charge - node_list[i].discharge)
            * (child_node.elapsed_time.total_seconds() / 3600)
            * child_node.coefficient
            * node_list[i].price
            for i in range(len(node_list))
            for child_node in node_list[i].next
        ),
        GRB.MAXIMIZE,
    )

    # Constraints
    lmps.generate_constraints(model, battery, initial_soc, final_soc)

    # Solve the model
    model.setParam(GRB.Param.Threads, 0)
    start_time = time.time()
    model.optimize()
    end_time = time.time()

    if model.Status == 2:
        return {
            "status_num": model.Status,
            "objective_value": model.objVal,
            "runtime": end_time - start_time,
            "model": model,
            "lmp_timeseries": lmps,
        }
    else:
        return {
            "status_num": model.Status,
            "lmp_timeseries": lmps,
        }
