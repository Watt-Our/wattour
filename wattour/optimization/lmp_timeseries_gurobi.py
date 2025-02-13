from typing import NamedTuple, Optional
from gurobipy import Model, Var

from wattour.core.battery import BatteryBase
from wattour.core.lmp import LMP
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase

class LMPNodeDecisionVars(NamedTuple):
    soe: Var
    charge: Optional[Var]
    discharge: Optional[Var]

# hacky structure
class LMPTimeseriesGurobi(LMPTimeseriesBase):
    def __init__(self, cls: LMPTimeseriesBase):
        super().__init__()
        self.timeseries = cls.timeseries
        self.lmp_decisions_vars = {}  # nodes are the keys, namedtuples are the values

    def add_gurobi_vars(self, model):
        """Add gurobi decision variables to each node."""
        if self.timeseries.head is None:
            raise ValueError("Timeseries is empty")

        def add_gurobi_vars_helper(model: Model, node: LMP):
            if node.dummy:
                decision_vars = LMPNodeDecisionVars(soe=model.addVar())
            else:
                decision_vars = LMPNodeDecisionVars(
                    soe=model.addVar(),
                    charge=model.addVar(),
                    discharge=model.addVar(),
                )
            self.lmp_decisions_vars[node.id] = decision_vars
            for child_node in node.next:
                add_gurobi_vars_helper(model, child_node)

        add_gurobi_vars_helper(model, self.timeseries.head)

    def generate_constraints(
        self, model: Model, battery: BatteryBase, initial_soc: float = 0, min_final_soc: float = 0
    ):
        """Generate constraints for a gurobi optimization problem."""
        if self.timeseries.head is None:
            raise ValueError("Timeseries is empty")

        max_soe = battery.get_usable_capacity()
        max_charge = battery.get_charge_rate()
        max_discharge = battery.get_discharge_rate()
        charge_eff = battery.get_charge_efficiency()
        discharge_eff = battery.get_discharge_efficiency()

        def generate_constraints_helper(node: LMP):
            # constraints
            model.addConstr(self.lmp_decisions_vars[node.id].soe <= max_soe)
            if node.dummy:
                model.addConstr(self.lmp_decisions_vars[node.id].soe >= min_final_soc * max_soe)
                return
            model.addConstr(self.lmp_decisions_vars[node.id].soe >= 0)
            model.addConstr(self.lmp_decisions_vars[node.id].charge <= max_charge)
            model.addConstr(self.lmp_decisions_vars[node.id].charge >= 0)
            model.addConstr(self.lmp_decisions_vars[node.id].discharge <= max_discharge)
            model.addConstr(self.lmp_decisions_vars[node.id].discharge >= 0)
            for child_node in node.next:
                if child_node.elapsed_time is None:
                    continue

                model.addConstr(
                    self.lmp_decisions_vars[child_node.id].soe
                    == self.lmp_decisions_vars[node.id].soe
                    + (
                        (
                            self.lmp_decisions_vars[node.id].charge * charge_eff
                            - self.lmp_decisions_vars[node.id].discharge / discharge_eff
                        )
                        - self.lmp_decisions_vars[node.id].soe * battery.get_self_discharge_rate()
                    )
                    * (child_node.elapsed_time.total_seconds() / 3600)
                )
                generate_constraints_helper(child_node)

        model.addConstr(self.lmp_decisions_vars[self.timeseries.head.id].soe == initial_soc * max_soe)
        generate_constraints_helper(self.timeseries.head)
