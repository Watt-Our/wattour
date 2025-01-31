from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

from gurobipy import Var


# helper class for individual nodes (linked list / graph)
# TODO: we should make as many params required as possible
@dataclass
class LMPBase:
    timestamp: Optional[datetime.datetime] = None
    elapsed_time: Optional[datetime.timedelta] = None  # time that elapsed between previous node and this one

    price: Optional[float] = None  # presumably $ / MW
    coefficient: Optional[float] = None  # Coefficient to multiply price by (for stochastic with several branches)
    charge: Optional[Var] = None
    discharge: Optional[Var] = None
    soc: Optional[Var] = None

    next: list[LMPBase] = field(default_factory=list)
    dummy: bool = False  # dummy nodes are used to represent time elapsed between last forecasted price
