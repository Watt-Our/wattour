from typing import Optional

from gurobipy import Var

from wattour.core import LMPBase


class LMPGurobi(LMPBase):
    # TODO: make this so that it overloads the constructor (being lazy now)

    charge: Optional[Var] = None
    discharge: Optional[Var] = None
    soc: Optional[Var] = None

    def create_from_lmp_base(self, lmp_base: LMPBase):
        self.timestamp = lmp_base.timestamp
        self.price = lmp_base.price
        self.elapsed_time = lmp_base.elapsed_time
        self.coefficient = lmp_base.coefficient
        self.next = lmp_base.next
        self.dummy = lmp_base.dummy
        self.charge = None
        self.discharge = None
        self.soc = None
        return self
