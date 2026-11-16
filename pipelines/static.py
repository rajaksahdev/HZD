"""Static features per ward: terrain, distance to drains, land cover. They change rarely, so this
runs by hand, not on the schedule.

Sources (no account needed):
  Copernicus DEM GLO-30  30 m surface heights. It is a surface model: in dense areas the heights
                         include buildings, so slope is noisier than true ground slope.
  ESA WorldCover 2021    10 m land cover.
  OpenStreetMap          rivers, streams, canals, drains and ditches, via the Overpass API.
                         Coverage of storm drains is incomplete; most are underground and unmapped.

python -m pipelines.static            download what is missing, compute, save
python -m pipelines.static --no-db    skip the database, write the CSV only
"""

import argparse
import json
import sys
from datetime import UTC, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from scipy import ndimage
from shapely.geometry import LineString

from api.config import settings
from pipelines import store
from scripts.load_wards import PATH as WARDS_PATH
from scripts.load_wards import ward_id_from_name

MARGIN = 0.015  # degrees around the wards, so edge cells have neighbours
METRIC_CRS = "EPSG:32643"  # UTM zone 43N, for lengths and areas in metres
LOW_LYING_M = 5.0  # height below which land is counted as low-lying

DEM_TILES = ["N18_00_E072_00", "N19_00_E072_00"]
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif"
COVER_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N18E072_Map.tif"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
WATERWAYS = "river|stream|canal|drain|ditch"

COVER_GROUPS = {  # WorldCover class codes
    "built_up_frac": [50],
    "green_frac": [10, 20, 30, 40],  # trees, shrubs, grass, crops
    "water_frac": [80],
    "wetland_frac": [90, 95],  # marsh and mangroves
}


def dem_path():
    return settings.raw_dir / "dem" / "copdem30_mumbai.tif"


def cover_path():
    return settings.raw_dir / "worldcover" / "worldcover_2021_mumbai.tif"


def waterways_path():
    return settings.raw_dir / "osm" / "waterways.json"


def features_path():
    return settings.processed_dir / "static" / "ward_features.csv"


def load_wards() -> gpd.GeoDataFrame:
    if not WARDS_PATH.exists():
        raise FileNotFoundError(f"{WARDS_PATH} not found. Run: python -m scripts.load_wards")
    wards = gpd.read_file(WARDS_PATH)
    wards["ward_id"] = wards["name"].map(ward_id_from_name)
    wards["geometry"] = wards.geometry.make_valid()
    return wards[["ward_id", "geometry"]].sort_values("ward_id").reset_index(drop=True)


def _bounds(wards: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    west, south, east, north = wards.total_bounds
    return west - MARGIN, south - MARGIN, east + MARGIN, north + MARGIN


def _read_window(url: str, bounds) -> tuple[np.ndarray, dict]:
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MAX_RETRY="4",
        GDAL_HTTP_RETRY_DELAY="10",
        GDAL_HTTP_USERAGENT=store.USER_AGENT,
    )
    with env, rasterio.open(f"/vsicurl/{url}") as src:
        b = src.bounds
        clipped = (
            max(bounds[0], b.left),
            max(bounds[1], b.bottom),
            min(bounds[2], b.right),
            min(bounds[3], b.top),
        )
        window = from_bounds(*clipped, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=window)
        profile = {
            "driver": "GTiff",
            "dtype": data.dtype,
            "count": 1,
            "crs": src.crs,
            "nodata": src.nodata,
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": window_transform(window, src.transform),
            "compress": "deflate",
        }
    return data, profile


def _save(path, data: np.ndarray, profile: dict, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    tmp.replace(path)
    store.log_download(path.parent.name, datetime.now(UTC).date(), path, url)


def fetch(wards: gpd.GeoDataFrame) -> None:
    """Download the pilot window of each source. Files already on disk are kept."""
    bounds = _bounds(wards)
    if not dem_path().exists():
        parts = []
        for tile in DEM_TILES:
            data, profile = _read_window(DEM_URL.format(t=tile), bounds)
            mem = rasterio.MemoryFile()
            with mem.open(**profile) as dst:
                dst.write(data, 1)
            parts.append(mem.open())
        mosaic, transform = merge(parts)
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
        _save(dem_path(), mosaic[0], profile, DEM_URL.format(t="{" + ",".join(DEM_TILES) + "}"))
    if not cover_path().exists():
        data, profile = _read_window(COVER_URL, bounds)
        _save(cover_path(), data, profile, COVER_URL)
    if not waterways_path().exists():
        west, south, east, north = bounds
        query = (
            f'[out:json][timeout:120];way["waterway"~"^({WATERWAYS})$"]'
            f"({south},{west},{north},{east});out geom;"
        )
        resp = store.http_get(OVERPASS_URL, params={"data": query}, timeout=180)
        path = waterways_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        store.log_download("osm", datetime.now(UTC).date(), path, OVERPASS_URL)


def load_waterways() -> gpd.GeoDataFrame:
    elements = json.loads(waterways_path().read_text(encoding="utf-8"))["elements"]
    lines = [
        LineString([(p["lon"], p["lat"]) for p in e["geometry"]])
        for e in elements
        if e["type"] == "way" and len(e.get("geometry", [])) >= 2
    ]
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


def cell_size_m(transform, lat: float) -> tuple[float, float]:
    """(east-west, north-south) size of one raster cell in metres at this latitude."""
    metres_per_degree = 111_320.0
    return (
        abs(transform.a) * metres_per_degree * np.cos(np.radians(lat)),
        abs(transform.e) * metres_per_degree,
    )


def slope_degrees(heights: np.ndarray, dx: float, dy: float) -> np.ndarray:
    d_row, d_col = np.gradient(heights.astype("float64"), dy, dx)
    return np.degrees(np.arctan(np.hypot(d_row, d_col)))


def distance_to_lines(lines: gpd.GeoDataFrame, shape, transform, dx: float, dy: float):
    """Distance in metres from every cell to the nearest line. NaN everywhere if no lines."""
    if lines.empty:
        return np.full(shape, np.nan)
    burned = features.rasterize(
        ((geom, 1) for geom in lines.geometry),
        out_shape=shape,
        transform=transform,
        all_touched=True,
        dtype="uint8",
    )
    if not burned.any():
        return np.full(shape, np.nan)
    return ndimage.distance_transform_edt(burned == 0, sampling=(dy, dx))


def ward_labels(wards: gpd.GeoDataFrame, shape, transform) -> np.ndarray:
    """Raster of ward numbers (1-based, in the order of `wards`); 0 outside every ward."""
    return features.rasterize(
        ((geom, i + 1) for i, geom in enumerate(wards.geometry)),
        out_shape=shape,
        transform=transform,
        dtype="int16",
    )


def build(wards: gpd.GeoDataFrame) -> pd.DataFrame:
    metric = wards.to_crs(METRIC_CRS)
    out = pd.DataFrame({"ward_id": wards["ward_id"], "area_km2": metric.area.to_numpy() / 1e6})
    index = np.arange(1, len(wards) + 1)

    with rasterio.open(dem_path()) as src:
        heights = src.read(1).astype("float64")
        transform, shape = src.transform, src.shape
        centre_lat = (src.bounds.top + src.bounds.bottom) / 2
    dx, dy = cell_size_m(transform, centre_lat)
    labels = ward_labels(wards, shape, transform)
    slope = slope_degrees(heights, dx, dy)
    waterways = load_waterways()
    distance = distance_to_lines(waterways, shape, transform, dx, dy)

    out["elevation_mean_m"] = ndimage.mean(heights, labels, index)
    out["elevation_p10_m"] = [np.percentile(heights[labels == i], 10) for i in index]
    out["low_lying_frac"] = ndimage.mean(heights < LOW_LYING_M, labels, index)
    out["slope_mean_deg"] = ndimage.mean(slope, labels, index)
    out["drain_distance_mean_m"] = ndimage.mean(distance, labels, index)

    lines = waterways.to_crs(METRIC_CRS)
    clipped = gpd.overlay(lines, metric[["ward_id", "geometry"]], how="intersection")
    km = clipped.length.groupby(clipped["ward_id"]).sum() / 1000
    out["drain_km_per_km2"] = out["ward_id"].map(km).fillna(0).to_numpy() / out["area_km2"]

    with rasterio.open(cover_path()) as src:
        cover = src.read(1)
        cover_labels = ward_labels(wards, src.shape, src.transform)
    for name, codes in COVER_GROUPS.items():
        out[name] = ndimage.mean(np.isin(cover, codes), cover_labels, index)
    return out.round(4)


def save_to_db(table: pd.DataFrame) -> None:
    from sqlalchemy.dialects.postgresql import insert

    from api.db import WardStatic, engine, init_db

    init_db()
    rows = table.to_dict("records")
    stmt = insert(WardStatic).values(rows)
    update = {c: stmt.excluded[c] for c in table.columns if c != "ward_id"}
    with engine.begin() as conn:
        conn.execute(stmt.on_conflict_do_update(index_elements=["ward_id"], set_=update))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute static features per ward.")
    parser.add_argument("--no-db", action="store_true", help="write the CSV only")
    args = parser.parse_args(argv)

    wards = load_wards()
    fetch(wards)
    table = build(wards)
    path = features_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    if not args.no_db:
        save_to_db(table)
    print(table.to_string(index=False))
    print(f"\n{len(table)} wards written to {path}" + ("" if args.no_db else " and to PostGIS"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
