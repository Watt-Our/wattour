from __future__ import annotations

import datetime

import pandas as pd
import pandera as pa
from pandas.api.types import is_numeric_dtype
from pandera.typing import Series

from wattour.core.utils.smth import Idk

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


class LMPTimeseriesBase:
    def __init__(self):
        self.timeseries: Idk[LMP] = Idk()

    def create_branch_from_df(self, lmp_df: pd.DataFrame, add_dummy: bool = True) -> None:
        """Populate the lmptimeseries from a dataframe (must be single link).

        Dataframe format must be [timestamp, lmp]. Returns the final node in the branch.
        """
        LMPDataFrame.validate(lmp_df)
        if lmp_df.empty:
            raise ValueError("The lmp_df DataFrame has no rows.")

        for _, row in lmp_df.iterrows():
            self.timeseries.append(LMP(timestamp=row["timestamp"], price=row["price"]))

        if add_dummy and self.timeseries.tail:
            self.timeseries.append_dummy(
                LMP(price=0, timestamp=self.timeseries.tail.timestamp + datetime.timedelta(hours=1))
            )

    def calc_coefficients(self):
        """Calculate coefficients based on branching to prevent overweighting timesteps with lots of branches."""
        if self.timeseries.head is None:
            raise ValueError("Timeseries is empty")

        # helper function for calc coefficients that calculates coefficients
        # for a node's children and then recursively calls the function for children nodes
        def calc_coefficients_helper(node: LMP):
            if node.next and node.coefficient:
                child_coefficient = node.coefficient / len(node.next)
                for child_node in node.next:
                    child_node.coefficient = child_coefficient
                    calc_coefficients_helper(child_node)

        self.timeseries.head.coefficient = 1.0
        calc_coefficients_helper(self.timeseries.head)

    # TODO: this should be moved down
    def get_node_list(self, dummies: bool) -> list:
        """Create a list of all node objects."""
        if self.timeseries.head is None:
            return []

        def get_node_list_helper(node: LMP, node_list: list):
            if node.dummy and (not dummies):
                return
            node_list.append(node)
            for child_node in node.next:
                get_node_list_helper(child_node, node_list)

        node_list = []
        get_node_list_helper(self.timeseries.head, node_list)
        return node_list

    # TODO: this too prob
    def __str__(self):
        """Return a string representation of the LMPTimeseriesBase instance."""
        return f"LMPTimeseriesBase: {self.timeseries.size} nodes, \
            {self.timeseries.branches} branches, {self.timeseries.dummy_nodes} dummy nodes"
