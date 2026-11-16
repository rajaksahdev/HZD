"""CHIRPS v3 daily rainfall (0.05 degree). Direct download, no account. Used for training labels.

The global file is large, so only the pilot window (plus a margin) is read from the server and
saved. The saved raw file is therefore a crop of the original, with values untouched.

We use the "sat" variant (within-pentad timing from IMERG). CHIRPS daily values are a
disaggregation of 5-day totals, so single-day amounts are less reliable than 5-day sums.
Files appear roughly a month after the date.
"""

import contextlib
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform

from api.config import settings
from pipelines import store
from pipelines.grid import empty_dataset, regrid

NAME = "chirps"
BASE = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/sat"
MARGIN = 0.25  # degrees around the pilot box


def _url(day: date) -> str:
    return f"{BASE}/{day:%Y}/chirps-v3.0.sat.{day:%Y.%m.%d}.tif"


def _name(day: date) -> str:
    return f"chirps-v3.0.sat.{day:%Y.%m.%d}.pilot.tif"


def fetch(day: date, force: bool = False) -> list[Path]:
    path = store.raw_dir(NAME, day) / _name(day)
    if path.exists():
        return []
    min_lon, min_lat, max_lon, max_lat = settings.pilot_bbox
    url = _url(day)
    env = contextlib.ExitStack()
    env.enter_context(
        rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MAX_RETRY="4",
            GDAL_HTTP_RETRY_DELAY="10",
            GDAL_HTTP_USERAGENT=store.USER_AGENT,
        )
    )
    try:
        remote = env.enter_context(rasterio.open(f"/vsicurl/{url}"))
    except rasterio.errors.RasterioIOError as exc:
        env.close()
        if "404" in str(exc):
            return []  # not published yet
        raise
    with env, remote as src:
        window = (
            from_bounds(
                min_lon - MARGIN,
                min_lat - MARGIN,
                max_lon + MARGIN,
                max_lat + MARGIN,
                src.transform,
            )
            .round_offsets()
            .round_lengths()
        )
        data = src.read(1, window=window)
        profile = src.profile | {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": window_transform(window, src.transform),
            "compress": "deflate",
        }
        for key in ("blockxsize", "blockysize", "tiled"):
            profile.pop(key, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    tmp.replace(path)
    store.log_download(NAME, day, path, url)
    return [path]


def harmonize(day: date) -> xr.Dataset | None:
    path = store.raw_dir(NAME, day) / _name(day)
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        data[data < 0] = np.nan
        t = src.transform
        lons = t.c + t.a * (np.arange(src.width) + 0.5)
        lats = t.f + t.e * (np.arange(src.height) + 0.5)
    ds = empty_dataset([np.datetime64(day.isoformat())])
    grid = regrid(data, lons, lats, method="average")
    ds["precipitation"] = (
        ("time", "lat", "lon"),
        grid[None, :, :],
        {"units": "mm", "accumulation": "1 day starting at `time` (UTC)"},
    )
    ds.attrs = {"source": NAME, "raw_file": path.name, "time_zone": "UTC"}
    return ds
