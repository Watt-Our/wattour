from gurobipy import Model

from wattour.core.battery import BatteryBase
from wattour.core.gurobi_lmp import GurobiLMP
from wattour.core.lmp_timeseries_base import LMPTimeseriesBase


# hacky structure
class LMPTimeseriesGurobi(LMPTimeseriesBase[GurobiLMP]):
    def __init__(self):
        super().__init__(GurobiLMP)

    def add_gurobi_vars(self, model: Model):
        """Add gurobi decision variables to each node."""
        if self.timeseries.head is None:
            raise ValueError("Timeseries is empty")

        for node in self.timeseries.iter_nodes():
            if node.dummy:
                node.soe = model.addVar()
            else:
                node.soe = model.addVar()
                node.charge = model.addVar()
                node.discharge = model.addVar()

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

        def generate_constraints_helper(node: GurobiLMP):
            # constraints
            model.addConstr(node.soe <= max_soe)
            if node.dummy:
                model.addConstr(node.soe >= min_final_soc * max_soe)
                return

            # for typing, for now
            assert node.charge
            assert node.discharge

            model.addConstr(node.soe >= 0)
            model.addConstr(node.charge <= max_charge)
            model.addConstr(node.charge >= 0)
            model.addConstr(node.discharge <= max_discharge)
            model.addConstr(node.discharge >= 0)
            for child_node in node.next:
                if child_node.elapsed_time is None:
                    continue

                model.addConstr(
                    child_node.soe
                    == node.soe
                    + (
                        (node.charge * charge_eff - node.discharge / discharge_eff)
                        - node.soe * battery.get_self_discharge_rate()
                    )
                    * (child_node.elapsed_time.total_seconds() / 3600)
                )
                generate_constraints_helper(child_node)

        model.addConstr(self.timeseries.head.soe == initial_soc * max_soe)
        generate_constraints_helper(self.timeseries.head)
