from __future__ import annotations

import datetime
from typing import Optional, overload

import pandas as pd
import pandera as pa
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
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, head: LMP) -> None:
        ...

    def __init__(self, head: Optional[LMP] = None) -> None:
        super().__init__()
        if head is not None:
            self.head = head
            self.size = 1
            self.branches = 1
        else:
            self.head = None
            self.size = 0
            self.branches = 0

    def create_branch_from_df(self, lmp_df: pd.DataFrame, add_dummy: bool = True) -> None:
        """Populate the lmptimeseries from a dataframe (must be single link).

        Dataframe format must be [timestamp, lmp]. Returns the final node in the branch.
        """
        LMPDataFrame.validate(lmp_df)
        if lmp_df.empty:
            raise ValueError("The lmp_df DataFrame has no rows.")

        prev_node = None
        for _, row in lmp_df.iterrows():
            if prev_node is None:
                self.head = LMP(timestamp=row["timestamp"], price=row["price"])
                prev_node = self.head
                continue
            cur_node = LMP(timestamp=row["timestamp"], price=row["price"])
            self.append(prev_node, cur_node)
            prev_node = cur_node

        if add_dummy:
            self.append_dummy(prev_node, prev_node.elapsed_time)

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

    def append_dummy(self, existing_node: LMP, elapsed_time: datetime.timedelta) -> None:
        """Append a dummy node to the existing node."""
        super().append_dummy(existing_node, LMP(price=0, timestamp=existing_node.timestamp + elapsed_time, is_dummy=True))

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
