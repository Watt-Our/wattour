from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional

from wattour.core.utils.tree import Node


# helper class for individual nodes (linked list / graph)
# TODO: we should make as many params required as possible
@dataclass
class LMP(Node["LMP"]):
    price: float  # presumably $ / MW
    timestamp: datetime.datetime
    elapsed_time: Optional[datetime.timedelta] = None  # time that elapsed between previous node and this one
    coefficient: Optional[float] = None  # coefficient to weight given node (important in optimization)

    def __post_init__(self):
        """Init Node parent."""
        super().__init__()

    def validate(self, prev: LMP) -> None:
        if not self.timestamp or not prev.timestamp:
            raise ValueError("All nodes must have a timestamp.")
        if self.timestamp <= prev.timestamp:
            raise ValueError("The new_node timestamp must be greater than the prev_node timestamp.")
        if prev.dummy:
            raise ValueError("Cannot add a node to a dummy node.")

    def enrich(self, prev: LMP) -> None:
        self.elapsed_time = self.timestamp - prev.timestamp
