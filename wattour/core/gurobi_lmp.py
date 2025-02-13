from typing import Optional

from gurobipy import Var

from .lmp import LMP


class GurobiLMP(LMP):
    soe: Var
    charge: Optional[Var] = None
    discharge: Optional[Var] = None
