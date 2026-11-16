"""IMD gridded daily rainfall (0.25 degree) through imdlib. No account needed.

Finished years come as one file per year; the current year comes as one real-time file per day.
imdlib decides the file names, so the raw layout here is raw/imd/yearly and raw/imd/realtime
rather than YYYY/MM/DD.

IMD's rainfall day ends at 03:00 UTC (08:30 IST) on the stated date.
"""

import contextlib
import io
from datetime import UTC, date, datetime
from pathlib import Path

import imdlib
import numpy as np
import xarray as xr

from api.config import settings
from pipelines import store
from pipelines.grid import empty_dataset, regrid

NAME = "imd"
YEARLY = settings.raw_dir / NAME / "yearly"
REALTIME = settings.raw_dir / NAME / "realtime"


def _is_yearly(day: date) -> bool:
    return day.year < datetime.now(UTC).year


def _files(folder: Path) -> set[Path]:
    return {p for p in folder.rglob("*") if p.is_file()}


def _open(day: date) -> xr.Dataset | None:
    quiet = contextlib.redirect_stdout(io.StringIO())
    try:
        with quiet:
            if _is_yearly(day):
                obj = imdlib.open_data("rain", day.year, day.year, "yearwise", str(YEARLY))
            else:
                obj = imdlib.open_real_data("rain", day.isoformat(), day.isoformat(), str(REALTIME))
            return obj.get_xarray()
    except (FileNotFoundError, OSError, ValueError):
        return None


def fetch(day: date, force: bool = False) -> list[Path]:
    if _open(day) is not None:
        return []
    folder = YEARLY if _is_yearly(day) else REALTIME
    folder.mkdir(parents=True, exist_ok=True)
    before = _files(folder)
    with contextlib.redirect_stdout(io.StringIO()):
        if _is_yearly(day):
            imdlib.get_data("rain", day.year, day.year, fn_format="yearwise", file_dir=str(folder))
        else:
            imdlib.get_real_data("rain", day.isoformat(), day.isoformat(), file_dir=str(folder))
    new = sorted(_files(folder) - before)
    for path in new:
        store.log_download(NAME, day, path, "imdlib (imdpune.gov.in)")
    return new


def harmonize(day: date) -> xr.Dataset | None:
    src = _open(day)
    if src is None:
        return None
    stamp = np.datetime64(day.isoformat())
    if stamp not in src["time"].values.astype("datetime64[D]"):
        return None
    field = src["rain"].sel(time=day.isoformat()).squeeze()
    field = field.transpose("lat", "lon")
    data = field.values.astype("float32")
    data[data < 0] = np.nan  # -999 marks missing
    min_lon, min_lat, max_lon, max_lat = settings.pilot_bbox
    lons, lats = field["lon"].values, field["lat"].values
    keep_lon = (lons >= min_lon - 1) & (lons <= max_lon + 1)
    keep_lat = (lats >= min_lat - 1) & (lats <= max_lat + 1)
    grid = regrid(data[np.ix_(keep_lat, keep_lon)], lons[keep_lon], lats[keep_lat], "bilinear")
    ds = empty_dataset([stamp])
    ds["precipitation"] = (
        ("time", "lat", "lon"),
        grid[None, :, :],
        {"units": "mm", "accumulation": "24 h ending 03:00 UTC on the date in `time`"},
    )
    ds.attrs = {"source": NAME, "time_zone": "UTC"}
    return ds
