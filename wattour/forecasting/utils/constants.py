# https://apiportal.pjm.com/api-details#api=9eb1ee98-d2bd-4ca0-a1c9-4a0df3413160&operation=da_hrl_lmps-Search

from typing import Literal

ALLOWED_ZONES = [
    "AECO",
    "AEP",
    "APS",
    "ATSI",
    "BGE",
    "COMED",
    "CPL",
    "DAY",
    "DEOK",
    "DOM",
    "DPL",
    "DUKE",
    "DUQ",
    "EKPC",
    "EXTERNAL",
    "JCPL",
    "METED",
    "PECO",
    "PENELEC",
    "PEPCO",
    "PPL",
    "PSEG",
    "RECO",
    "OVEC",
]

COMMON_LMP_ALLOWED_FIELDS = [
    "datetime_beginning_utc",
    "pnode_id",
    "pnode_name",
    "voltage",
    "equipment",
    "type",
    "zone",
]

DA_LMP_ALLOWED_FIELDS = [
    "system_energy_price_da",
    "total_lmp_da",
    "congestion_price_da",
    "marginal_loss_price_da",
]

RT_LMP_ALLOWED_FIELDS = [
    "system_energy_price_rt",
    "total_lmp_rt",
    "congestion_price_rt",
    "marginal_loss_price_rt",
]
BATCH_SIZE = 50_000

# [date-value] to [date-value]. The date-value range should be within 366 days.
# The date-value should include the date and time component.
# Example: yyyy-MM-dd HH:mm to yyyy-MM-dd HH:mm.
BEGIN_DATE_ALLOWED_VALUES = Literal[
    "Today",
    "CurrentHour",
    "CurrentWeek",
    "CurrentMonth",
    "CurrentYear",
    "Yesterday",
    "LastHour",
    "LastWeek",
    "LastMonth",
    "LastYear",
    "Tomorrow",
    "NextWeek",
    "NextMonth",
    "NextYear",
    "15SecondsAgo",
    "5MinutesAgo",
    "1MonthAgo",
    "4MonthsAgo",
    "6MonthsAgo",
]
