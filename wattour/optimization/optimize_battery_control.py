import time
from typing import Any, NamedTuple, Optional

import gurobipy as gp
from gurobipy import GRB, Model, Var

from wattour.core import BatteryBase
from wattour.core.lmp import LMP
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase

from .lmp_timeseries_gurobi import LMPTimeseriesGurobi


class BatteryControlResult(NamedTuple):
    status_num: int
    lmp_timeseries: LMPTimeseriesGurobi
    objective_value: Optional[Any] = None
    runtime: Optional[float] = None
    model: Optional[gp.Model] = None

class LMPDecisionVariables(NamedTuple):
    soe: float
    charge: float
    discharge: float

def __create_gurobi_vars(timeseries: LMPTimeseriesBase, model: Model):
    """Add gurobi decision variables to each node.

    Returns: dict with nodes as keys and decision tuple as values
    """
    if timeseries.head is None:
        raise ValueError("Timeseries is empty")

    decisions_vars = {}
    for node in timeseries.get_node_list():
        if node.dummy:
            decision_var = LMPDecisionVariables(soe=model.addVar())
        else:
            decision_var = LMPDecisionVariables(
                soe=model.addVar(),
                charge=model.addVar(),
                discharge=model.addVar(),
            )
        decisions_vars[node] = decision_var

    return decisions_vars

def __generate_constraints(
    timeseries: LMPTimeseriesBase, decision_vars: dict, model: Model, battery: BatteryBase, initial_soc: float = 0, min_final_soc: float = 0
):
    """Generate constraints for a gurobi optimization problem."""
    if timeseries.head is None:
        raise ValueError("Timeseries is empty")

    max_soe = battery.get_usable_capacity()
    max_charge = battery.get_charge_rate()
    max_discharge = battery.get_discharge_rate()
    charge_eff = battery.get_charge_efficiency()
    discharge_eff = battery.get_discharge_efficiency()

    def generate_constraints_helper(node: LMP):
        # constraints
        model.addConstr(decision_vars[node].soe <= max_soe)
        if node.dummy:
            model.addConstr(decision_vars[node].soe >= min_final_soc * max_soe)
            return

        # for typing, for now
        assert isinstance(decision_vars[node].charge, Var), "charge is not a Var"
        assert isinstance(decision_vars[node].discharge, Var), "discharge is not a Var"

        model.addConstr(decision_vars[node].soe >= 0)
        model.addConstr(decision_vars[node].charge <= max_charge)
        model.addConstr(decision_vars[node].charge >= 0)
        model.addConstr(decision_vars[node].discharge <= max_discharge)
        model.addConstr(decision_vars[node].discharge >= 0)
        for child_node in node.next:
            if child_node.elapsed_time is None:
                continue

            model.addConstr(
                decision_vars[child_node].soe
                == decision_vars[node].soe
                + (
                    (decision_vars[node].charge * charge_eff - decision_vars[node].discharge / discharge_eff)
                    - decision_vars[node].soe * battery.get_self_discharge_rate()
                )
                * (decision_vars[child_node].elapsed_time.total_seconds() / 3600)
            )
            generate_constraints_helper(child_node)

    model.addConstr(timeseries.head.soe == initial_soc * max_soe)
    generate_constraints_helper(timeseries.head)


# LMPTimeseries has branches, this function will complete stochastic optimization
def optimize_battery_control(
    battery: BatteryBase, lmps: LMPTimeseriesBase, initial_soc: float = 0, final_soc: float = 0
) -> BatteryControlResult:
    if lmps.head is None:
        raise ValueError("Timeseries is empty")

    # Check that initial and final state of charge are valid
    if initial_soc > 1 or initial_soc < 0:
        raise ValueError("Invalid initial state of charge")
    if final_soc > 1 or final_soc < 0:
        raise ValueError("Invalid final state of charge")

    model = gp.Model("Battery Control Optimizer")

    if lmps.head.coefficient is None:
        lmps.calc_coefficients()

    decision_vars = __create_gurobi_vars(lmps, model)
    node_list = lmps.get_node_list(show_dummy=False)

    # Objective function; charge and dischare are in power units
    model.setObjective(
        gp.quicksum(
            (decision_vars[node_list[i]].discharge - decision_vars[node_list[i]].charge)
            * (child_node.elapsed_time.total_seconds() / 3600)
            * child_node.coefficient
            * node_list[i].price
            for i in range(len(node_list))
            for child_node in node_list[i].next
        ),
        GRB.MAXIMIZE,
    )

    # Constraints
    __generate_constraints(lmps, decision_vars, model, battery, initial_soc, final_soc)

    # Solve the model
    model.setParam(GRB.Param.Threads, 0)
    start_time = time.time()
    model.optimize()
    end_time = time.time()

    if model.Status == 2:
        return BatteryControlResult(
            status_num=model.Status,
            objective_value=model.objVal,
            runtime=end_time - start_time,
            model=model,
            lmp_timeseries=lmps,
        )
    else:
        return BatteryControlResult(
            status_num=model.Status,
            lmp_timeseries=lmps,
        )
