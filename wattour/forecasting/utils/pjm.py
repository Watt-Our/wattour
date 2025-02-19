import datetime
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import dotenv
import pandas as pd
import requests

from .constants import BATCH_SIZE, BEGIN_DATE_ALLOWED_VALUES, COMMON_LMP_ALLOWED_FIELDS, RT_LMP_ALLOWED_FIELDS

dotenv.load_dotenv()


class PJMError(Exception):
    pass


PJM_API = "https://api.pjm.com/api/v1"
PJM_API_KEY = os.environ.get("PJM_API_KEY", None)
if PJM_API_KEY is None:
    raise OSError("No API key provided")

PJM_RATE_LIMIT = 6  # reqs/sec


# TODO: put this somewhere else
# logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def create_csv(func: Callable[..., pd.DataFrame], output_file_path: str):
    def helper(*args, **kwargs):
        df = func(*args, **kwargs)
        df.to_csv(output_file_path, index=False)
        csv_size_mb = Path(output_file_path).stat().st_size / 1024 / 1024
        logging.info(f"Outputted file to {output_file_path} with size {csv_size_mb:.1f} MB")

    return helper


def get_lmps(
    base_req_url: str,
    start_time: BEGIN_DATE_ALLOWED_VALUES,
    lmp_allowed_fields: list[str],
    zone: Optional[str] = None,
    **kwargs,
):
    """Fetch LMP data from PJM API in batches."""
    # LastYear, PSEG returns 6642349 rows - with batches of 50k this is ~120 requests, at 6req/min for ~20 minutes
    params = {
        "download": False,
        "datetime_beginning_utc": start_time,
        "fields": ",".join(lmp_allowed_fields),
        **kwargs.get("params", {}),
    }
    if zone is not None:
        params["zone"] = zone

    sleep_rate_limit = 60 / PJM_RATE_LIMIT

    # need rowCount (max 50k) and startRow (1-indexed)
    data_rows = []
    start_row = 1
    total_rows = None
    while True:
        cur_params = {**params, "startRow": start_row, "rowCount": BATCH_SIZE}
        req_url = f"{base_req_url}?{'&'.join(f'{k}={v}' for k, v in cur_params.items())}"

        try:
            r = requests.get(req_url, timeout=30, headers={"Ocp-Apim-Subscription-Key": PJM_API_KEY})
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(r)
            logging.exception("Request failed")

            if r.status_code == 429:
                time.sleep(30)
                continue

            if r.status_code != 200:
                raise PJMError(f"Status code was not 200, but {r.status_code}") from e

        res = r.json()
        total_rows = res.get("totalRows", 0)
        data_rows.extend(res.get("items", []))

        if start_row + BATCH_SIZE > total_rows:
            break
        else:
            start_row = (start_row + BATCH_SIZE) % total_rows

        time.sleep(sleep_rate_limit)

    return pd.DataFrame(data_rows)


# make this more abstract
def get_node_fivemin(pnode_id: str) -> pd.DataFrame:
    """Get a node's rt 5min LMP data for a specified time period."""
    base_req_url = f"{PJM_API}/rt_fivemin_hrl_lmps"

    params = {
        "download": False,
        "pnode_id": pnode_id,
        "datetime_beginning_utc": "CurrentHour",
        "fields": ",".join(COMMON_LMP_ALLOWED_FIELDS + RT_LMP_ALLOWED_FIELDS),
    }

    df = get_lmps(
        base_req_url,
        "Today",
        COMMON_LMP_ALLOWED_FIELDS + RT_LMP_ALLOWED_FIELDS,
        params=params,
    )
    if df.empty:
        raise PJMError("No data received for given node.")

    return df


# TODO: this function should get the latest available (unverified) lmp price for a given node
# and return a tuple with (datetime, price)
def get_latest_price(pnode_id: str):
    return datetime.now(), 0.0
