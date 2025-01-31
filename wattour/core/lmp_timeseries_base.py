from __future__ import annotations

import datetime

import pandas as pd

from .lmp_base import LMPBase


# class to represent lmp timeseries
# TODO: If we need more basic lmptimeseries, main functionality should be moved to abstract and gurobi
# specific functions should be moved to a child class
class LMPTimeseriesBase:
    def __init__(self, head: LMPBase):
        self.head = head
        self.total_nodes: int = 1
        self.branches: int = 1
        self.dummy_nodes: int = 0  # can use this to check that all branches have a dummy node

    def add_node(self, prev_node: LMPBase, new_node: LMPBase) -> None:
        """Add a node to another node (linked list)."""
        if not new_node.timestamp or not prev_node.timestamp:
            raise ValueError("The new_node and prev_node must have a timestamp.")
        if new_node.timestamp <= prev_node.timestamp:
            raise ValueError("The new_node timestamp must be greater than the prev_node timestamp.")
        if prev_node.dummy:
            raise ValueError("Cannot add a node to a dummy node.")

        new_node.elapsed_time = new_node.timestamp - prev_node.timestamp
        if prev_node.next:
            self.branches += 1
        prev_node.next.append(new_node)
        self.total_nodes += 1

    def create_branch_from_df(self, start_node: LMPBase, lmp_df: pd.DataFrame) -> None:
        """Populate the lmptimeseries from a dataframe (must be single link).

        Dataframe format must be [timestamp, lmp]. Returns the final node in the branch.
        """
        if lmp_df.empty:
            raise ValueError("The lmp_df DataFrame has no rows.")

        prev_node = start_node
        for _, row in lmp_df.iterrows():
            current_node = LMPBase(timestamp=row["timestamp"], price=row["lmp"])
            self.add_node(prev_node, current_node)

        return current_node

    def add_dummy_node(self, node: LMPBase, elapsed_time: datetime.timedelta) -> None:
        """Add dummy node to the end of timseries branch."""
        new_timestamp = node.timestamp + elapsed_time if node.timestamp else None
        new_node = LMPBase(timestamp=new_timestamp, price=0, elapsed_time=elapsed_time)
        new_node.dummy = True

        if node.next:
            self.branches += 1
        self.dummy_nodes += 1
        node.next.append(new_node)

    def calc_coefficients(self):
        """Calculate coefficients based on branching to prevent overweighting timesteps with lots of branches."""

        # helper function for calc coefficients that calculates coefficients
        # for a node's children and then recursively calls the function for children nodes
        def calc_coefficients_helper(node: LMPBase):
            if node.next and node.coefficient:
                child_coefficient = node.coefficient / len(node.next)
                for child_node in node.next:
                    child_node.coefficient = child_coefficient
                    calc_coefficients_helper(child_node)

        self.head.coefficient = 1.0
        calc_coefficients_helper(self.head)

    # create a list of all node objects
    def get_node_list(self, dummies: bool) -> list:
        def get_node_list_helper(node: LMPBase, node_list: list):
            if node.dummy and (not dummies):
                return
            node_list.append(node)
            for child_node in node.next:
                get_node_list_helper(child_node, node_list)

        node_list = []
        get_node_list_helper(self.head, node_list)
        return node_list
