"""Hourly sea level (tide) near Mumbai harbour from the Open-Meteo marine API. No account needed.

This is a model estimate, not a tide-gauge reading. High tide blocks the storm drains, so the
flood model uses it as an input.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from pipelines import store

NAME = "tide"
URL = "https://marine-api.open-meteo.com/v1/marine"
LAT, LON = 18.92, 72.83  # off Colaba / Mumbai harbour
VAR = "sea_level_height_msl"


def fetch(day: date, force: bool = False) -> list[Path]:
    is_past = day < datetime.now(UTC).date()
    if is_past and not force and store.latest_raw(NAME, day, "sea_level_*.json"):
        return []
    resp = store.http_get(
        URL,
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": VAR,
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "timezone": "UTC",
        },
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return [store.write_raw(NAME, day, f"sea_level_{stamp}.json", resp.content, resp.url)]


def harmonize(day: date) -> xr.Dataset | None:
    path = store.latest_raw(NAME, day, "sea_level_*.json")
    if path is None:
        return None
    hourly = json.loads(path.read_text(encoding="utf-8"))["hourly"]
    values = np.array([np.nan if v is None else v for v in hourly[VAR]], dtype="float32")
    ds = xr.Dataset(
        {"sea_level": (("time",), values, {"units": "m", "datum": "mean sea level"})},
        coords={"time": np.asarray(hourly["time"], dtype="datetime64[ns]")},
    )
    ds.attrs = {"source": NAME, "raw_file": path.name, "time_zone": "UTC", "lat": LAT, "lon": LON}
    return ds
