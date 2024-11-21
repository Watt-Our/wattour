import time
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from .battery import Battery

# Optimize battery control for given battery and lmps
# lmp format: [timestamp, lmp]
def optimize_battery_control(battery: Battery, lmps: pd.DataFrame, initial_soc=0, final_soc=0):
    t_steps = len(lmps) # total number of time steps    
    
    # Battery parameters
    max_c = battery.get_charge_rate() # maximum charging rate
    max_d = battery.get_discharge_rate() # maximum discharging rate
    charge_eff = battery.get_charge_efficiency() # charging efficiency
    discharge_eff = battery.get_discharge_efficiency() # discharging efficiency
    self_discharge_rate = battery.get_self_discharge_rate() # self discharge rate (% per hour)
    max_soc = battery.get_usable_capacity() # maximum state of charge
    
    # Check that initial and final state of charge are valid
    if initial_soc > max_soc or initial_soc < 0:
        raise ValueError("Invalid initial state of charge")
    if final_soc > max_soc or final_soc < 0:
        raise ValueError("Invalid final state of charge")
    

    # Account for time step durations
    if not pd.api.types.is_datetime64_any_dtype(lmps['timestamp']):
        raise ValueError("The 'timestamp' column must be of datetime type")
    lmps['time_step_duration'] = lmps['timestamp'].diff().dt.total_seconds() / 3600

    model = gp.Model("Battery Control Optimizer")

    # Variables
    c = model.addVars(t_steps - 1)  # Charging at each time step (in energy)
    d = model.addVars(t_steps - 1)  # Discharging at each time step (in energy)
    s = model.addVars(t_steps)  # State of charge at each time step (in energy units)

    # Maximizing the profit
    model.setObjective(
        gp.quicksum(lmps['lmp'][t] * (d[t - 1] - c[t - 1] * lmps['time_step_duration'][t]) for t in range(1, t_steps)), GRB.MAXIMIZE 
    )

    # Constraints
    model.addConstr(s[0] == initial_soc)
    model.addConstrs(
        (
            s[t]
            == s[t - 1]
            + charge_eff * c[t - 1]
            - (1 / discharge_eff) * d[t - 1]
            - lmps['time_step_duration'][t] * self_discharge_rate * s[t - 1]
            for t in range(1, t_steps)
        ),
        "cSOC",
    )

    model.addConstrs((c[t] >= 0 for t in range(0, t_steps - 1)), "cCNonNeg")
    model.addConstrs((c[t - 1] <= max_c * lmps['time_step_duration'][t] for t in range(1, t_steps)), "cMaxC")
    model.addConstrs((d[t] >= 0 for t in range(0, t_steps - 1)), "cDNonNeg")
    model.addConstrs((d[t - 1] <= max_d * lmps['time_step_duration'][t] for t in range(1, t_steps)), "cMaxD")
    model.addConstrs((s[t] >= 0 for t in range(0, t_steps)), "cSOCNonNeg")
    model.addConstrs((s[t] <= max_soc for t in range(0, t_steps)), "cMaxSOC")
    model.addConstr(s[t_steps - 1] == final_soc)

    model.setParam(GRB.Param.Threads, 0)
    start_time = time.time()
    model.optimize()
    end_time = time.time()
        
    if model.Status == 2:
        charge_values = [c[i].X for i in range(t_steps - 1)]
        discharge_values = [d[i].X for i in range(t_steps - 1)]
        soc_values = [s[i].X for i in range(t_steps)]
        return {
            "status_num": model.Status,
            "message": "Problem solved optimally",
            "objective_value": model.objVal,
            "runtime": end_time - start_time,
            "charge_values": charge_values,
            "discharge_values": discharge_values,
            "soc_values": soc_values,
        }
    else:
        return {
            "status_num": model.Status,
            "message": "Problem did not optimize",
        }