from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from gurobipy import Model, Var

from .battery import Battery


# helper class for individual nodes (linked list / graph)
# TODO: we should make as many params required as possible
@dataclass
class LMP:
    timestamp: Optional[datetime.datetime] = None
    elapsed_time: Optional[datetime.timedelta] = None  # time that elapsed between previous node and this one

    price: Optional[float] = None  # presumably $ / MW
    coefficient: Optional[float] = None  # Coefficient to multiply price by (for stochastic with several branches)
    charge: Optional[Var] = None
    discharge: Optional[Var] = None
    soc: Optional[Var] = None

    next: list[LMP] = field(default_factory=list)
    dummy: bool = False  # dummy nodes are used to represent time elapsed between last forecasted price


# class to represent lmp timeseries
# TODO: If we need more basic lmptimeseries, main functionality should be moved to abstract and gurobi
# specific functions should be moved to a child class
class LMPTimeseriesBase:
    def __init__(self, head: LMP):
        self.head = head
        self.total_nodes = 1
        self.branches = 1

    def add_node(self, prev_node: LMP, new_node: LMP) -> None:
        """Add a node to another node (linked list)."""
        if not new_node.timestamp or not prev_node.timestamp:
            return

        new_node.elapsed_time = new_node.timestamp - prev_node.timestamp
        if prev_node.next:
            self.branches += 1
        prev_node.next.append(new_node)
        self.total_nodes += 1

    def create_from_df(self, lmp_df: pd.DataFrame, final_time_interval: datetime.timedelta) -> None:
        """Populate the lmptimeseries from a dataframe (must be single link) dataframe format must be [timestamp, lmp].

        Final elapsed time is the time between the last timestamp and the end of the optimization period.
        """
        prev_node = None
        for _, row in lmp_df.iterrows():
            current_node = LMP(timestamp=row["timestamp"], price=row["lmp"])
            if prev_node is not None:
                self.add_node(prev_node, current_node)
            else:
                self.head = current_node
            prev_node = current_node

        if prev_node is not None:
            self.add_dummy_node(prev_node, final_time_interval)

    def add_dummy_node(self, node: LMP, elapsed_time: datetime.timedelta) -> None:
        """Add dummy node to the end of timseries branch."""
        new_timestamp = node.timestamp + elapsed_time if node.timestamp else None
        new_node = LMP(timestamp=new_timestamp, price=0, elapsed_time=elapsed_time)
        new_node.dummy = True

        if node.next:
            self.branches += 1
        node.next.append(new_node)

    def calc_coefficients(self):
        """Calculate coefficients based on branching to prevent overweighting timesteps with lots of branches."""

        # helper function for calc coefficients that calculates coefficients
        # for a node's children and then recursively calls the function for children nodes
        def calc_coefficients_helper(node: LMP):
            if node.next and node.coefficient:
                child_coefficient = node.coefficient / len(node.next)
                for child_node in node.next:
                    child_node.coefficient = child_coefficient
                    calc_coefficients_helper(child_node)

        self.head.coefficient = 1.0
        calc_coefficients_helper(self.head)

    def add_gurobi_vars(self, model):
        """Add gurobi decision variables to each node."""

        def add_gurobi_vars_helper(model: Model, node: LMP):
            node.soc = model.addVar()
            if node.dummy:
                return
            node.charge = model.addVar()
            node.discharge = model.addVar()
            for child_node in node.next:
                add_gurobi_vars_helper(model, child_node)

        add_gurobi_vars_helper(model, self.head)

    # create a list of all node objects
    def get_node_list(self, dummies: bool) -> list:
        def get_node_list_helper(node: LMP, node_list: list):
            if node.dummy and (not dummies):
                return
            node_list.append(node)
            for child_node in node.next:
                get_node_list_helper(child_node, node_list)

        node_list = []
        get_node_list_helper(self.head, node_list)
        return node_list

    # generate constraints for a gurobi optimization problem
    def generate_constraints(self, model: Model, battery: Battery, initial_soc=0, min_final_soc=0):
        # helper function to generate constraints for each node
        max_soc = battery.get_usable_capacity()
        max_charge = battery.get_charge_rate()
        max_discharge = battery.get_discharge_rate()
        charge_eff = battery.get_charge_efficiency()
        discharge_eff = battery.get_discharge_efficiency()

        def generate_constraints_helper(node: LMP):
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


class LMPTimeseriesGurobi(LMPTimeseriesBase):
    pass
