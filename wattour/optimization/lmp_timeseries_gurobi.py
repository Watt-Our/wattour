from gurobipy import Model

from wattour.core.battery import BatteryBase
from wattour.core.lmp import LMP
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase


class LMPTimeseriesGurobi(LMPTimeseriesBase):
    def __init__(self, head: LMP):
        super().__init__(head)  # Call parent __init__
        self.lmp_decisions_vars = {}  # nodes are the keys, dictionary of decision variables

    def add_gurobi_vars(self, model):
        """Add gurobi decision variables to each node."""

        def add_gurobi_vars_helper(model: Model, node: LMP):
            if node.dummy:
                decision_vars = {"soc": model.addVar()}
            else:
                decision_vars = {"soc": model.addVar(), "charge": model.addVar(), "discharge": model.addVar()}
            self.lmp_decisions_vars[node.id] = decision_vars
            for child_node in node.next:
                add_gurobi_vars_helper(model, child_node)

        add_gurobi_vars_helper(model, self.head)

    # generate constraints for a gurobi optimization problem
    def generate_constraints(
        self, model: Model, battery: BatteryBase, initial_soc: float = 0, min_final_soc: float = 0
    ):
        # helper function to generate constraints for each node
        max_soc = battery.get_usable_capacity()
        max_charge = battery.get_charge_rate()
        max_discharge = battery.get_discharge_rate()
        charge_eff = battery.get_charge_efficiency()
        discharge_eff = battery.get_discharge_efficiency()

        def generate_constraints_helper(node: LMP):
            # constraints
            model.addConstr(self.lmp_decisions_vars[node.id]["soc"] <= max_soc)
            if node.dummy:
                model.addConstr(self.lmp_decisions_vars[node.id]["soc"] >= min_final_soc)
                return
            model.addConstr(self.lmp_decisions_vars[node.id]["soc"] >= 0)
            model.addConstr(self.lmp_decisions_vars[node.id]["charge"] <= max_charge)
            model.addConstr(self.lmp_decisions_vars[node.id]["charge"] >= 0)
            model.addConstr(self.lmp_decisions_vars[node.id]["discharge"] <= max_discharge)
            model.addConstr(self.lmp_decisions_vars[node.id]["discharge"] >= 0)
            for child_node in node.next:
                if child_node.elapsed_time is None:
                    continue

                model.addConstr(
                    self.lmp_decisions_vars[child_node.id]["soc"]
                    == self.lmp_decisions_vars[node.id]["soc"]
                    + (
                        (
                            self.lmp_decisions_vars[node.id]["charge"] * charge_eff
                            - self.lmp_decisions_vars[node.id]["discharge"] / discharge_eff
                        )
                        - self.lmp_decisions_vars[node.id]["soc"] * battery.get_self_discharge_rate()
                    )
                    * (child_node.elapsed_time.total_seconds() / 3600)
                )
                generate_constraints_helper(child_node)

        model.addConstr(self.lmp_decisions_vars[self.head.id]["soc"] == initial_soc)
        generate_constraints_helper(self.head)
