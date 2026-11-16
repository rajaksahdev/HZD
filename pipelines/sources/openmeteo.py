"""Open-Meteo multi-model forecast on the pilot grid. No account needed (non-commercial use).

For today and future days this is a real forecast. For past days the API returns each model's
stitched short-lead forecasts, which is what we bias-correct against observations in Stage 4.
Each fetch is saved under its own timestamp, so newer forecasts never overwrite older ones.
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from pipelines import store
from pipelines.grid import empty_dataset, target_coords

NAME = "openmeteo"
RECENT_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
RECENT_DAYS = 80  # the live endpoint keeps roughly three months of past days
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
VARIABLES = {  # API name -> unit after harmonising
    "precipitation": "mm",
    "temperature_2m": "degC",
    "relative_humidity_2m": "%",
    "wind_speed_10m": "m s-1",
}


def _points() -> tuple[list[float], list[float]]:
    lons, lats = target_coords()
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    return grid_lat.ravel().tolist(), grid_lon.ravel().tolist()


def fetch(day: date, force: bool = False) -> list[Path]:
    today = datetime.now(UTC).date()
    is_past = day < today
    if is_past and not force and store.latest_raw(NAME, day, "forecast_*.json"):
        return []  # a finished day does not change
    lats, lons = _points()
    url = RECENT_URL if day >= today - timedelta(days=RECENT_DAYS) else ARCHIVE_URL
    resp = store.http_get(
        url,
        params={
            "latitude": ",".join(f"{v:.2f}" for v in lats),
            "longitude": ",".join(f"{v:.2f}" for v in lons),
            "hourly": ",".join(VARIABLES),
            "models": ",".join(MODELS),
            "wind_speed_unit": "ms",
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "timezone": "UTC",
        },
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return [store.write_raw(NAME, day, f"forecast_{stamp}.json", resp.content, resp.url)]


def harmonize(day: date) -> xr.Dataset | None:
    path = store.latest_raw(NAME, day, "forecast_*.json")
    if path is None:
        return None
    points = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(points, dict):
        points = [points]
    lons, lats = target_coords()
    if len(points) != lats.size * lons.size:
        raise ValueError(f"{path}: expected {lats.size * lons.size} points, got {len(points)}")

    ds = empty_dataset(points[0]["hourly"]["time"])
    ds = ds.assign_coords(model=MODELS)
    shape = (len(MODELS), ds.sizes["time"], lats.size, lons.size)
    for var, unit in VARIABLES.items():
        cube = np.full(shape, np.nan, dtype="float32")
        for m, model in enumerate(MODELS):
            for p, point in enumerate(points):  # same order as _points(): lat-major
                values = point["hourly"].get(f"{var}_{model}")
                if values is not None:
                    cube[m, :, p // lons.size, p % lons.size] = np.array(values, dtype="float32")
        ds[var] = (("model", "time", "lat", "lon"), cube, {"units": unit})
    ds.attrs = {"source": NAME, "raw_file": path.name, "time_zone": "UTC"}
    return ds
