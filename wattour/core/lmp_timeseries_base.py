from __future__ import annotations

import datetime

import pandas as pd
import pandera as pa
from pandas.api.types import is_numeric_dtype
from pandera.typing import Series

from .lmp import LMP


class LMPDataFrame(pa.DataFrameModel):
    price: Series
    timestamp: Series[pd.Timestamp]

    # temporary, until decide float or int
    @pa.check("price")
    def check_is_number(self, column_header: Series):
        return is_numeric_dtype(column_header)


def transform(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    new_df = df[column_map.keys()].rename(columns=column_map)
    LMPDataFrame.validate(new_df)
    return new_df


# class to represent lmp timeseries
# TODO: If we need more basic lmptimeseries, main functionality should be moved to abstract and gurobi
# specific functions should be moved to a child class
class LMPTimeseriesBase:
    def __init__(self, head: LMP, nodes: int, branches: int, dummy_nodes: int):
        self.head = head
        self.total_nodes: int = nodes
        self.branches: int = branches
        self.dummy_nodes: int = dummy_nodes  # can use this to check that all branches have a dummy node

    @staticmethod
    def add_node(prev_node: LMP, new_node: LMP) -> None:
        """Add a node to another node (linked list)."""
        if not new_node.timestamp or not prev_node.timestamp:
            raise ValueError("The new_node and prev_node must have a timestamp.")
        if new_node.timestamp <= prev_node.timestamp:
            raise ValueError("The new_node timestamp must be greater than the prev_node timestamp.")
        if prev_node.dummy:
            raise ValueError("Cannot add a node to a dummy node.")

        new_node.elapsed_time = new_node.timestamp - prev_node.timestamp
        prev_node.next.append(new_node)

    @staticmethod
    def add_dummy_node(node: LMP, elapsed_time: datetime.timedelta) -> None:
        """Add dummy node to the end of timseries branch."""
        new_timestamp = node.timestamp + elapsed_time if node.timestamp else None
        if not new_timestamp:
            return

        new_node = LMP(timestamp=new_timestamp, price=0, elapsed_time=elapsed_time)
        new_node.dummy = True
        node.next.append(new_node)

    # this can be refactored so that the dt is decoupled from instantiation w/ an ADT that stores info about struct
    @staticmethod
    def create_branch_from_df(lmp_df: pd.DataFrame):
        """Populate the lmptimeseries from a dataframe (must be single link).

        Dataframe format must be [timestamp, lmp]. Returns the final node in the branch.
        """
        print(lmp_df)
        LMPDataFrame.validate(lmp_df)
        if lmp_df.empty:
            raise ValueError("The lmp_df DataFrame has no rows.")

        total_nodes = 0
        branches = 0
        dummy_nodes = 0

        prev_node = None
        for _, row in lmp_df.iterrows():
            current_node = LMP(timestamp=row["timestamp"], price=row["price"])
            if not prev_node:
                head = current_node
            else:
                LMPTimeseriesBase.add_node(prev_node, current_node)
                if prev_node.next:
                    branches += 1

            total_nodes += 1
            prev_node = current_node

        LMPTimeseriesBase.add_dummy_node(current_node, datetime.timedelta(hours=1))
        if current_node.next:
            branches += 1
        dummy_nodes += 1

        return LMPTimeseriesBase(head, total_nodes, branches, dummy_nodes)

    def add_branch(self, node: LMP, branch: LMPTimeseriesBase):
        """Add a branch to a node."""
        if node.dummy:
            raise ValueError("Cannot add a branch to a dummy node.")

        if not node.next:
            self.branches += 1
        self.branches += branch.branches - 1
        self.total_nodes += branch.total_nodes
        self.dummy_nodes += branch.dummy_nodes

        node.next.append(branch.head)

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

    def __str__(self):
        """Return a string representation of the LMPTimeseriesBase instance."""
        return f"LMPTimeseriesBase: {self.total_nodes} nodes, {self.branches} branches, {self.dummy_nodes} dummy nodes"
