from __future__ import annotations

import datetime
from typing import Optional, Self

import pandas as pd
import pandera as pa
from matplotlib import pyplot as plt
from pandas.api.types import is_numeric_dtype
from pandera.typing import Series

from wattour.core.utils.tree import Tree

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


class LMPTimeseriesBase(Tree[LMP]):
    def __init__(self) -> None:
        super().__init__()

    def create_branch_from_df(
        self, lmp_df: pd.DataFrame, add_dummy: bool = True, on_node: Optional[LMP] = None
    ) -> Self:
        """Populate the lmptimeseries from a dataframe (must be single link).

        Dataframe format must be [timestamp, lmp]. Returns the final node in the branch.
        """
        LMPDataFrame.validate(lmp_df)
        if lmp_df.empty:
            raise ValueError("The lmp_df DataFrame has no rows.")

        prev_node = on_node if on_node else self.head
        for _, row in lmp_df.iterrows():
            cur_node = LMP(timestamp=row["timestamp"], price=row["price"])
            self.append(prev_node, cur_node)
            prev_node = cur_node

        if add_dummy and prev_node:
            if not prev_node.elapsed_time:
                raise ValueError("Previous node does not have an elapsed time")

            self.append_dummy(
                prev_node, LMP(price=0, timestamp=prev_node.timestamp + prev_node.elapsed_time, is_dummy=True)
            )

        return self

    def calc_coefficients(self):
        """Calculate coefficients based on branching to prevent overweighting timesteps with lots of branches."""
        if self.head is None:
            raise ValueError("Timeseries is empty")

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

    def weight_coefficients(self, weight: float) -> None:
        """Multiply the coefficients of the nodes by a weight."""
        if self.head is None:
            raise ValueError("Timeseries is empty")

        for node in self.iter_nodes():
            if node.coefficient:
                node.coefficient *= weight

    def get_node_list(self, show_dummy: bool = True) -> list[LMP]:
        """Create a list of all node objects."""
        if self.head is None:
            return []
        return list(self.iter_nodes(show_dummy))

    def plot(self) -> None:
        """Plot the timeseries with connections between each parent and child node."""
        if self.head is None:
            raise ValueError("Timeseries is empty")

        def plot_node(
            node: LMP, parent_timestamp: Optional[datetime.datetime] = None, parent_price: Optional[float] = None
        ):
            if parent_timestamp is not None and parent_price is not None:
                plt.plot([parent_timestamp, node.timestamp], [parent_price, node.price], "b-")  # type: ignore
            for child in node.next:
                if not child.dummy:
                    plot_node(child, node.timestamp, node.price)

        plot_node(self.head)
        plt.xticks(rotation=90)
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.show()
