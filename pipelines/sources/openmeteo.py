"""Open-Meteo multi-model forecast on the pilot grid. No account needed (non-commercial use).

For today and future days this is a real forecast. For past days the API returns each model's
stitched short-lead forecasts, which is what we bias-correct against observations in Stage 4.
Each fetch is saved under its own timestamp, so newer forecasts never overwrite older ones.

Days older than the live endpoint keeps are fetched a calendar month at a time (one request
instead of thirty) and saved under raw/openmeteo/monthly/YYYY/MM/.
"""

import calendar
import json
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

from api.config import settings
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


def _monthly_file(day: date) -> Path | None:
    folder = settings.raw_dir / NAME / "monthly" / f"{day:%Y}" / f"{day:%m}"
    files = sorted(folder.glob("forecast_month_*.json"))
    return files[-1] if files else None


def _request(url: str, start: date, end: date):
    lats, lons = _points()
    return store.http_get(
        url,
        params={
            "latitude": ",".join(f"{v:.2f}" for v in lats),
            "longitude": ",".join(f"{v:.2f}" for v in lons),
            "hourly": ",".join(VARIABLES),
            "models": ",".join(MODELS),
            "wind_speed_unit": "ms",
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "UTC",
        },
        timeout=180,
    )


def fetch(day: date, force: bool = False) -> list[Path]:
    today = datetime.now(UTC).date()
    is_past = day < today
    have = store.latest_raw(NAME, day, "forecast_*.json") or _monthly_file(day)
    if is_past and not force and have:
        return []  # a finished day does not change
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if day < today - timedelta(days=RECENT_DAYS):
        first = day.replace(day=1)
        last = day.replace(day=calendar.monthrange(day.year, day.month)[1])
        resp = _request(ARCHIVE_URL, first, last)
        path = settings.raw_dir / NAME / "monthly" / f"{day:%Y}" / f"{day:%m}"
        path = path / f"forecast_month_{stamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
        store.log_download(NAME, first, path, resp.url)
        return [path]
    resp = _request(RECENT_URL, day, day)
    return [store.write_raw(NAME, day, f"forecast_{stamp}.json", resp.content, resp.url)]


@lru_cache(maxsize=2)
def _read(path: Path) -> list[dict]:
    points = json.loads(path.read_text(encoding="utf-8"))
    return [points] if isinstance(points, dict) else points


def harmonize(day: date) -> xr.Dataset | None:
    path = store.latest_raw(NAME, day, "forecast_*.json") or _monthly_file(day)
    if path is None:
        return None
    points = _read(path)
    lons, lats = target_coords()
    if len(points) != lats.size * lons.size:
        raise ValueError(f"{path}: expected {lats.size * lons.size} points, got {len(points)}")

    all_times = points[0]["hourly"]["time"]
    keep = [i for i, t in enumerate(all_times) if t.startswith(day.isoformat())]
    if not keep:
        return None
    hours = slice(keep[0], keep[-1] + 1)  # a monthly file holds many days; take this one
    ds = empty_dataset(all_times[hours])
    ds = ds.assign_coords(model=MODELS)
    shape = (len(MODELS), ds.sizes["time"], lats.size, lons.size)
    for var, unit in VARIABLES.items():
        cube = np.full(shape, np.nan, dtype="float32")
        for m, model in enumerate(MODELS):
            for p, point in enumerate(points):  # same order as _points(): lat-major
                values = point["hourly"].get(f"{var}_{model}")
                if values is not None:
                    cell = np.array(values[hours], dtype="float32")
                    cube[m, :, p // lons.size, p % lons.size] = cell
        ds[var] = (("model", "time", "lat", "lon"), cube, {"units": unit})
    ds.attrs = {"source": NAME, "raw_file": path.name, "time_zone": "UTC"}
    return ds
