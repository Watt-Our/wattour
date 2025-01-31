from gurobipy import Model

from wattour.core.battery import BatteryBase
from wattour.core.lmp_base import LMPBase
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase

from .lmp_gurobi import LMPGurobi


class LMPTimeseriesGurobi(LMPTimeseriesBase):
    def __init__(self, head: LMPGurobi | None = None):
        self.head = head
        self.total_nodes: int = 1
        self.branches: int = 1
        self.dummy_nodes: int = 0  # can use this to check that all branches have a dummy node

    def create_from_lmp_timeseries_base(self, lmp_timeseries_base: LMPTimeseriesBase):
        """Create a gurobi version of the lmp timeseries from a base lmp timeseries."""

        def create_from_lmp_base_helper(node: LMPBase):
            new_node = LMPGurobi(node) # will rely on fixing other functionality. 
            for child_node in node.next:
                new_node.next.append(create_from_lmp_base_helper(child_node))
            return new_node

        self.head = create_from_lmp_base_helper(lmp_timeseries_base.head)

    def add_gurobi_vars(self, model):
        """Add gurobi decision variables to each node."""

        def add_gurobi_vars_helper(model: Model, node: LMPGurobi):
            node.soc = model.addVar()
            if node.dummy:
                return
            node.charge = model.addVar()
            node.discharge = model.addVar()
            for child_node in node.next:
                add_gurobi_vars_helper(model, child_node)

        add_gurobi_vars_helper(model, self.head)

    # generate constraints for a gurobi optimization problem
    def generate_constraints(self, model: Model, battery: BatteryBase, initial_soc=0, min_final_soc=0):
        # helper function to generate constraints for each node
        max_soc = battery.get_usable_capacity()
        max_charge = battery.get_charge_rate()
        max_discharge = battery.get_discharge_rate()
        charge_eff = battery.get_charge_efficiency()
        discharge_eff = battery.get_discharge_efficiency()

        def generate_constraints_helper(node: LMPBase):
            if not (node.soc and node.charge and node.discharge):
                return

            # Constraints
            model.addConstr(node.soc <= max_soc)
            if node.dummy:
                model.addConstr(node.soc >= min_final_soc)
                return
            model.addConstr(node.soc >= 0)
            model.addConstr(node.charge <= max_charge)
            model.addConstr(node.charge >= 0)
            model.addConstr(node.discharge <= max_discharge)
            model.addConstr(node.discharge >= 0)
            for child_node in node.next:
                if child_node.elapsed_time is None:
                    continue

                model.addConstr(
                    child_node.soc
                    == node.soc
                    + (
                        (node.charge * charge_eff - node.discharge / discharge_eff)
                        - node.soc * battery.get_self_discharge_rate()
                    )
                    * (child_node.elapsed_time.total_seconds() / 3600)
                )
                generate_constraints_helper(child_node)

        model.addConstr(self.head.soc == initial_soc)
        generate_constraints_helper(self.head)
