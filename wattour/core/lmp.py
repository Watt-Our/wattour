from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Optional


# helper class for individual nodes (linked list / graph)
# TODO: we should make as many params required as possible
@dataclass
class LMP:
    timestamp: datetime.datetime
    price: float  # presumably $ / MW

    elapsed_time: Optional[datetime.timedelta] = None  # time that elapsed between previous node and this one
    coefficient: Optional[float] = None  # coefficient to weight given node (important in optimization)

    next: list[LMP] = field(default_factory=list)
    dummy: bool = False  # dummy nodes are used to represent time elapsed between last forecasted price
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

