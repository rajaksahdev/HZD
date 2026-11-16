"""The common pilot grid: 0.1 degree cell centres over PILOT_BBOX, time in UTC."""

import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from api.config import settings

RES = 0.1
CRS = "EPSG:4326"


def target_coords() -> tuple[np.ndarray, np.ndarray]:
    """(lons ascending, lats ascending) of cell centres."""
    min_lon, min_lat, max_lon, max_lat = settings.pilot_bbox
    lons = np.round(np.arange(min_lon, max_lon + RES / 2, RES), 4)
    lats = np.round(np.arange(min_lat, max_lat + RES / 2, RES), 4)
    return lons, lats


def _north_up_transform(lons: np.ndarray, lats_desc: np.ndarray):
    res_x = float(abs(lons[1] - lons[0]))
    res_y = float(abs(lats_desc[0] - lats_desc[1]))
    return from_origin(lons[0] - res_x / 2, lats_desc[0] + res_y / 2, res_x, res_y)


def regrid(data: np.ndarray, src_lons, src_lats, method: str = "bilinear") -> np.ndarray:
    """Regrid a 2-D (lat, lon) array on a regular lat/lon grid to the pilot grid.

    method: "average" when the source is finer than 0.1 degree (keeps rainfall totals),
            "bilinear" when it is coarser. NaN is treated as missing.
    Returns an array shaped (lat ascending, lon ascending).
    """
    src_lons = np.asarray(src_lons, dtype=float)
    src_lats = np.asarray(src_lats, dtype=float)
    src = np.asarray(data, dtype="float32")
    if src_lons[0] > src_lons[-1]:
        src_lons, src = src_lons[::-1], src[:, ::-1]
    if src_lats[0] < src_lats[-1]:  # make north-up
        src_lats, src = src_lats[::-1], src[::-1, :]

    lons, lats = target_coords()
    dst = np.full((lats.size, lons.size), np.nan, dtype="float32")
    reproject(
        source=np.ascontiguousarray(src),
        destination=dst,
        src_transform=_north_up_transform(src_lons, src_lats),
        src_crs=CRS,
        src_nodata=np.nan,
        dst_transform=_north_up_transform(lons, lats[::-1]),
        dst_crs=CRS,
        dst_nodata=np.nan,
        resampling=Resampling[method],
    )
    return dst[::-1, :]  # back to lat ascending


def empty_dataset(times) -> xr.Dataset:
    lons, lats = target_coords()
    return xr.Dataset(
        coords={"time": np.asarray(times, dtype="datetime64[ns]"), "lat": lats, "lon": lons}
    )
