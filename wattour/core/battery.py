from abc import ABC, abstractmethod


# Abstract class for batteries
class BatteryBase(ABC):
    # Return usable capacity in MWh
    @abstractmethod
    def get_usable_capacity(self) -> float:
        pass

    # Return maximum charging rate in MW
    @abstractmethod
    def get_charge_rate(self) -> float:
        pass

    # Return maximum discharging rate in MW
    @abstractmethod
    def get_discharge_rate(self) -> float:
        pass

    # Return charge efficiecy in % < 1
    @abstractmethod
    def get_charge_efficiency(self) -> float:
        pass

    # Return discharge efficiency in % < 1
    @abstractmethod
    def get_discharge_efficiency(self) -> float:
        pass

    # Return self discharge rate in % < 1 (% of initial energy lost per hour)
    @abstractmethod
    def get_self_discharge_rate(self) -> float:
        pass


# Generic battery class
class GenericBattery(BatteryBase):
    def __init__(
        self, usable_capacity, charge_rate, discharge_rate, charge_efficiency, discharge_efficiency, self_discharge_rate
    ):
        self.usable_capacity = usable_capacity
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.self_discharge_rate = self_discharge_rate

    def get_usable_capacity(self) -> float:
        return self.usable_capacity

    def get_charge_rate(self) -> float:
        return self.charge_rate

    def get_discharge_rate(self) -> float:
        return self.discharge_rate

    def get_charge_efficiency(self) -> float:
        return self.charge_efficiency

    def get_discharge_efficiency(self) -> float:
        return self.discharge_efficiency

    def get_self_discharge_rate(self) -> float:
        return self.self_discharge_rate
