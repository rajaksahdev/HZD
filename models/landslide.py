"""Landslide susceptibility (Stage 10): where landslides are likely, on 1 km cells over the
Mumbai Metropolitan Region and Raigad, scored on held-out areas.

Area       A box around MMR and Raigad district: 72.75-73.85 E, 17.80-19.80 N, cut into 0.01
           degree cells (about 1.1 x 1.05 km). Cells that are mostly water are dropped.
Inventory  NASA Global Landslide Catalog (open CSV, events from news reports, 2007-2017), events
           located to 1 km or better. The GSI inventory (Bhukosh portal) needs a login and was not
           reachable from here: put its export (GeoJSON, GeoPackage or shapefile, points or
           polygons) in data/raw/gsi/ and it is used as well, marked by source.
Features   Per cell, from Copernicus DEM (30 m): mean elevation, relief, mean and 90th percentile
           slope, share of slopes over 30 degrees. From ESA WorldCover (read at 40 m): built-up,
           tree, low vegetation and bare shares. From OpenStreetMap: distance to the nearest
           river or stream, and to the nearest main road or railway. From IMD: mean June-September
           rain over the years on disk.
Models     XGBoost and random forest against a slope-only baseline (rank cells by steep slope).
Split      Spatial: 0.25 degree blocks (about 27 km), five folds, each block held out once. A
           random split would score cells next to the ones the model was trained on.
Scores     PR-AUC (average precision) of the pooled held-out predictions, with a 90% range from
           resampling blocks, and recall: the share of held-out landslide cells in the top 10%
           and 20% most susceptible land of their fold.
Bias       A news-based inventory reports landslides that hit roads, railways and homes, so
           features that describe those learn where landslides get reported as much as where
           slopes fail. Road and rail distance was left out of the final model before any score
           was seen. Land cover was left out after the first run, when built-up and low
           vegetation shares turned out to be its two strongest features (decided from the
           feature importance, not the scores). Both variants are still scored and reported.
           The final model uses terrain, distance to rivers and monsoon rain only.

python -m models.landslide --fetch                      inventory, DEM, land cover, OSM lines
python -m models.landslide --report docs/landslide.md   build cells, score, save the final model
"""

import argparse
import json
import sys
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from shapely.geometry import LineString
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from api.config import settings
from pipelines import static, store
from pipelines.sources import imd

REGION = (72.75, 17.80, 73.85, 19.80)  # west, south, east, north
CELL = 0.01  # degrees
BLOCK = 0.25  # degrees, for the spatial split
FOLDS = 5
DEM_PX = 3600  # Copernicus GLO-30: 1 arc-second pixels
COVER_PX = 3000  # WorldCover read at a quarter of its 10 m resolution (1/12000 degree)
LINE_PX = 1000  # OSM lines are rasterised at 0.001 degree
STEEP_DEG = 30.0
TOP_SHARES = (0.10, 0.20)
DEM_TILES = [
    "N17_00_E073_00",
    "N18_00_E072_00",
    "N18_00_E073_00",
    "N19_00_E072_00",
    "N19_00_E073_00",
]
COVER_TILES = ["N15E072", "N18E072"]
COVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{t}_Map.tif"
)
GLC_URL = (
    "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/"
    "Global_Landslide_Catalog_Export_rows.csv"
)
GOOD_ACCURACY = ("exact", "1km")  # finer than or equal to one cell
COVER = {
    "built_frac": [50],
    "tree_frac": [10],
    "low_veg_frac": [20, 30, 40],  # shrubs, grass, crops
    "bare_frac": [60],
}
WATER = [80, 90, 95]  # open water, wetland, mangroves
TERRAIN = ["elev_mean_m", "relief_m", "slope_mean_deg", "slope_p90_deg", "steep_frac"]
LAND = list(COVER)
OTHER = ["river_km", "rain_jjas_mm"]
TRANSPORT = ["road_km"]
FEATURE_SETS = {
    "terrain": TERRAIN + OTHER,
    "with_cover": TERRAIN + LAND + OTHER,
    "with_roads": TERRAIN + LAND + OTHER + TRANSPORT,
}
FINAL = "terrain"

RAW = settings.raw_dir
OUT = settings.processed_dir / "landslide"
CELLS_PATH = OUT / "cells.csv"
MODEL_PATH = settings.processed_dir / "models" / "landslide_xgb.json"
META_PATH = settings.processed_dir / "models" / "landslide_xgb_meta.json"
RASTER_PATH = OUT / "susceptibility.tif"
WARDS_PATH = OUT / "ward_susceptibility.csv"


def dem_path() -> Path:
    return RAW / "dem" / "copdem30_region.tif"


def cover_path() -> Path:
    return RAW / "worldcover" / "worldcover_2021_region_40m.tif"


def lines_path() -> Path:
    return RAW / "osm" / "region_lines.json"


def glc_path() -> Path | None:
    files = sorted((RAW / "glc").glob("glc_*.csv"))
    return files[-1] if files else None


def grid_shape() -> tuple[int, int]:
    west, south, east, north = REGION
    return round((north - south) / CELL), round((east - west) / CELL)


# --- Downloads ------------------------------------------------------------------------------


def fetch_glc() -> Path:
    if glc_path():
        return glc_path()
    resp = store.http_get(GLC_URL, timeout=300)
    path = RAW / "glc" / f"glc_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    store.log_download("glc", datetime.now(UTC).date(), path, GLC_URL)
    return path


def fetch_dem() -> None:
    if dem_path().exists():
        return
    parts = []
    for tile in DEM_TILES:
        url = static.DEM_URL.format(t=tile)
        try:
            data, profile = static._read_window(url, REGION)
        except RasterioIOError:
            print(f"  DEM tile {tile} not found (all sea?), skipped")
            continue
        mem = rasterio.MemoryFile()
        with mem.open(**profile) as dst:
            dst.write(data, 1)
        parts.append(mem.open())
    mosaic, transform = merge(parts, bounds=REGION, res=1 / DEM_PX, nodata=np.nan)
    profile.update(
        height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, nodata=np.nan
    )
    static._save(dem_path(), mosaic[0].astype("float32"), profile, static.DEM_URL)


def fetch_cover() -> None:
    """WorldCover at 40 m: each tile's part of the region, read from its overviews."""
    if cover_path().exists():
        return
    west, south, east, north = REGION
    rows, cols = round((north - south) * COVER_PX), round((east - west) * COVER_PX)
    out = np.zeros((rows, cols), dtype="uint8")  # 0 = no data
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4")
    for tile in COVER_TILES:
        url = COVER_URL.format(t=tile)
        with env, rasterio.open(f"/vsicurl/{url}") as src:
            b = src.bounds
            part = (max(west, b.left), max(south, b.bottom), min(east, b.right), min(north, b.top))
            if part[0] >= part[2] or part[1] >= part[3]:
                continue
            window = from_bounds(*part, src.transform).round_offsets().round_lengths()
            h, w = round((part[3] - part[1]) * COVER_PX), round((part[2] - part[0]) * COVER_PX)
            data = src.read(1, window=window, out_shape=(h, w), resampling=Resampling.nearest)
        r0, c0 = round((north - part[3]) * COVER_PX), round((part[0] - west) * COVER_PX)
        out[r0 : r0 + h, c0 : c0 + w] = data
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "crs": "EPSG:4326",
        "nodata": 0,
        "height": rows,
        "width": cols,
        "transform": from_origin(west, north, 1 / COVER_PX, 1 / COVER_PX),
        "compress": "deflate",
    }
    static._save(cover_path(), out, profile, COVER_URL.format(t="{" + ",".join(COVER_TILES) + "}"))


def fetch_lines() -> None:
    if lines_path().exists():
        return
    west, south, east, north = REGION
    box = f"({south},{west},{north},{east})"
    query = (
        "[out:json][timeout:300];("
        f'way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]{box};'
        f'way["railway"="rail"]{box};'
        f'way["waterway"~"^(river|stream)$"]{box};'
        ");out geom;"
    )
    resp = store.http_get(static.OVERPASS_URL, params={"data": query}, timeout=400)
    path = lines_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    store.log_download("osm", datetime.now(UTC).date(), path, static.OVERPASS_URL)


def fetch() -> None:
    print("inventory", fetch_glc().relative_to(RAW).as_posix(), flush=True)
    fetch_dem()
    print("DEM", dem_path().relative_to(RAW).as_posix(), flush=True)
    fetch_cover()
    print("land cover", cover_path().relative_to(RAW).as_posix(), flush=True)
    fetch_lines()
    print("OSM lines", lines_path().relative_to(RAW).as_posix(), flush=True)


# --- Inventory ------------------------------------------------------------------------------


def glc_events() -> pd.DataFrame:
    path = glc_path()
    if path is None:
        raise FileNotFoundError("No landslide catalog. Run: python -m models.landslide --fetch")
    raw = pd.read_csv(path)
    west, south, east, north = REGION
    inside = raw["longitude"].between(west, east) & raw["latitude"].between(south, north)
    rows = raw[inside].copy()
    return pd.DataFrame(
        {
            "date": pd.to_datetime(rows["event_date"], format="%m/%d/%Y %I:%M:%S %p").dt.date,
            "lat": rows["latitude"],
            "lon": rows["longitude"],
            "accuracy": rows["location_accuracy"].fillna("unknown"),
            "category": rows["landslide_category"],
            "title": rows["event_title"],
            "link": rows["source_link"],
            "source": "NASA GLC",
        }
    ).reset_index(drop=True)


def gsi_events() -> pd.DataFrame:
    """Any GSI export placed in data/raw/gsi/. Polygons become a point inside them."""
    frames = []
    for path in sorted((RAW / "gsi").glob("*")):
        if path.suffix.lower() not in (".geojson", ".json", ".gpkg", ".shp"):
            continue
        shapes = gpd.read_file(path).to_crs(4326)
        points = shapes.geometry.representative_point()
        frames.append(
            pd.DataFrame(
                {
                    "date": None,
                    "lat": points.y.values,
                    "lon": points.x.values,
                    "accuracy": "exact",
                    "category": "landslide",
                    "title": path.name,
                    "link": None,
                    "source": "GSI",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def events() -> pd.DataFrame:
    table = pd.concat([glc_events(), gsi_events()], ignore_index=True)
    west, south, east, north = REGION
    inside = table["lon"].between(west, east) & table["lat"].between(south, north)
    return table[inside].reset_index(drop=True)


def cell_of(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    west, _, _, north = REGION
    rows = np.floor((north - np.asarray(lat)) / CELL).astype(int)
    cols = np.floor((np.asarray(lon) - west) / CELL).astype(int)
    return rows, cols


# --- Cell features --------------------------------------------------------------------------


def _blocks(array: np.ndarray, size: int) -> np.ndarray:
    """(rows*size, cols*size) -> (rows, cols, size*size), cropping any remainder."""
    rows, cols = array.shape[0] // size, array.shape[1] // size
    a = array[: rows * size, : cols * size].reshape(rows, size, cols, size)
    return a.transpose(0, 2, 1, 3).reshape(rows, cols, size * size)


def _fit(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad or crop a per-cell array to the region grid."""
    out = np.full(shape, np.nan, dtype="float64")
    r, c = min(shape[0], array.shape[0]), min(shape[1], array.shape[1])
    out[:r, :c] = array[:r, :c]
    return out


def terrain_cells(shape) -> dict[str, np.ndarray]:
    with rasterio.open(dem_path()) as src:
        heights = src.read(1).astype("float32")
        heights[heights < -100] = np.nan
    size = round(CELL * DEM_PX)
    dx, dy = static.cell_size_m(from_origin(0, 0, 1 / DEM_PX, 1 / DEM_PX), np.mean(REGION[1::2]))
    slope = static.slope_degrees(np.nan_to_num(heights, nan=0.0), dx, dy).astype("float32")
    slope[np.isnan(heights)] = np.nan
    h, s = _blocks(heights, size), _blocks(slope, size)
    with np.errstate(all="ignore"):
        out = {
            "elev_mean_m": np.nanmean(h, axis=2),
            "relief_m": np.nanmax(h, axis=2) - np.nanmin(h, axis=2),
            "slope_mean_deg": np.nanmean(s, axis=2),
            "slope_p90_deg": np.nanpercentile(s, 90, axis=2),
            "steep_frac": np.nanmean(s > STEEP_DEG, axis=2),
        }
    return {k: _fit(v, shape) for k, v in out.items()}


def cover_cells(shape) -> dict[str, np.ndarray]:
    with rasterio.open(cover_path()) as src:
        cover = src.read(1)
    blocks = _blocks(cover, round(CELL * COVER_PX))
    known = (blocks > 0).sum(axis=2).astype("float64")
    known[known == 0] = np.nan
    out = {name: np.isin(blocks, codes).sum(axis=2) / known for name, codes in COVER.items()}
    out["water_frac"] = np.isin(blocks, WATER).sum(axis=2) / known
    return {k: _fit(v, shape) for k, v in out.items()}


def load_lines() -> dict[str, gpd.GeoSeries]:
    elements = json.loads(lines_path().read_text(encoding="utf-8"))["elements"]
    groups: dict[str, list] = {"river": [], "road": []}
    for e in elements:
        geometry = e.get("geometry", [])
        if e["type"] != "way" or len(geometry) < 2:
            continue
        tags = e.get("tags", {})
        group = "river" if "waterway" in tags else "road"
        groups[group].append(LineString([(p["lon"], p["lat"]) for p in geometry]))
    return {k: gpd.GeoSeries(v, crs=4326) for k, v in groups.items()}


def distance_cells(lines: gpd.GeoSeries, shape) -> np.ndarray:
    """Distance (km) from each cell centre to the nearest line, via a 0.001 degree raster."""
    west, south, east, north = REGION
    fine = (round((north - south) * LINE_PX), round((east - west) * LINE_PX))
    transform = from_origin(west, north, 1 / LINE_PX, 1 / LINE_PX)
    dx, dy = static.cell_size_m(transform, np.mean(REGION[1::2]))
    distance = static.distance_to_lines(gpd.GeoDataFrame(geometry=lines), fine, transform, dx, dy)
    step = round(CELL * LINE_PX)
    centres = distance[step // 2 :: step, step // 2 :: step] / 1000
    return _fit(centres, shape)


def rain_cells(shape) -> np.ndarray:
    """Mean June-September IMD rain (mm) over the yearly files on disk, at cell centres."""
    totals = []
    for path in sorted((imd.YEARLY / "rain").glob("*.grd")):
        year = int(path.stem)
        ds = imd._open(date(year, 7, 1))
        if ds is None:
            continue
        rain = ds["rain"].where(ds["rain"] >= 0)
        season = rain.sel(time=slice(f"{year}-06-01", f"{year}-09-30"))
        totals.append(season.sum("time", skipna=False).sel(lat=slice(17, 21), lon=slice(72, 75)))
    if not totals:
        raise FileNotFoundError("No IMD yearly rain. Run: python -m pipelines.run --sources imd")
    # Years can differ in the last digits of their coordinates; they share the 0.25 degree grid,
    # so average the values by position on the first year's coordinates.
    first = totals[0]
    if any(t.shape != first.shape for t in totals):
        raise ValueError("IMD yearly files cover different grids")
    stacked = np.stack([t.transpose("lat", "lon").values for t in totals])
    mean = first.transpose("lat", "lon").copy(data=stacked.mean(axis=0))
    mean = mean.assign_coords(
        lat=np.round(mean["lat"].values, 2), lon=np.round(mean["lon"].values, 2)
    )
    west, _, _, north = REGION
    lats = north - CELL * (np.arange(shape[0]) + 0.5)
    lons = west + CELL * (np.arange(shape[1]) + 0.5)
    smooth = mean.interp(lat=lats, lon=lons)
    nearest = mean.sel(lat=lats, lon=lons, method="nearest").assign_coords(lat=lats, lon=lons)
    filled = smooth.fillna(nearest)
    # sea cells have no IMD value; take the nearest land value along each row
    grid = pd.DataFrame(filled.transpose("lat", "lon").values)
    return grid.ffill(axis=1).bfill(axis=1).to_numpy(dtype="float64")


def build_cells() -> pd.DataFrame:
    shape = grid_shape()
    columns = {**terrain_cells(shape), **cover_cells(shape)}
    lines = load_lines()
    columns["river_km"] = distance_cells(lines["river"], shape)
    columns["road_km"] = distance_cells(lines["road"], shape)
    columns["rain_jjas_mm"] = rain_cells(shape)
    rows, cols = np.indices(shape)
    west, _, _, north = REGION
    table = pd.DataFrame({"row": rows.ravel(), "col": cols.ravel()})
    table["lat"] = (north - CELL * (table["row"] + 0.5)).round(4)
    table["lon"] = (west + CELL * (table["col"] + 0.5)).round(4)
    for name, values in columns.items():
        table[name] = values.ravel()
    table["block"] = (table["row"] // round(BLOCK / CELL)) * 100 + table["col"] // round(
        BLOCK / CELL
    )
    land = (table["water_frac"] < 0.5) & table["elev_mean_m"].notna()
    return table[land].reset_index(drop=True).round(4)


def cells(rebuild: bool = False) -> pd.DataFrame:
    if CELLS_PATH.exists() and not rebuild:
        return pd.read_csv(CELLS_PATH)
    table = build_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(CELLS_PATH, index=False)
    return table


def label(table: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    good = inventory[inventory["accuracy"].isin(GOOD_ACCURACY)]
    rows, cols = cell_of(good["lat"], good["lon"])
    hit = set(zip(rows.tolist(), cols.tolist(), strict=True))
    table = table.copy()
    table["landslide"] = [
        int((r, c) in hit) for r, c in zip(table["row"], table["col"], strict=True)
    ]
    return table


# --- Models and the spatial split ----------------------------------------------------------


def make_model(kind: str, positives: int, negatives: int):
    if kind == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            scale_pos_weight=negatives / max(positives, 1),
            eval_metric="aucpr",
            n_jobs=4,
            random_state=0,
        )
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=0,
    )


METHODS = {
    "xgb_terrain": ("xgboost", "terrain", "XGBoost, terrain and rain (final)"),
    "rf_terrain": ("random_forest", "terrain", "random forest, terrain and rain"),
    "xgb_cover": ("xgboost", "with_cover", "XGBoost, plus land cover"),
    "xgb_roads": ("xgboost", "with_roads", "XGBoost, plus land cover and road distance"),
    "slope": (None, None, "slope only (90th percentile slope)"),
}


def folds(table: pd.DataFrame) -> np.ndarray:
    """Fold number per cell; whole blocks go to one fold."""
    fold = np.zeros(len(table), dtype=int)
    for k, (_, test) in enumerate(GroupKFold(n_splits=FOLDS).split(table, groups=table["block"])):
        fold[test] = k
    return fold


def held_out(table: pd.DataFrame) -> pd.DataFrame:
    """Held-out scores for every method: each cell is scored by a model that never saw its block."""
    fold = folds(table)
    out = pd.DataFrame({"block": table["block"], "fold": fold, "landslide": table["landslide"]})
    for key, (kind, feature_set, _) in METHODS.items():
        scores = np.zeros(len(table))
        for k in range(FOLDS):
            test = fold == k
            if kind is None:
                scores[test] = table.loc[test, "slope_p90_deg"]
                continue
            train = table[~test]
            y = train["landslide"]
            model = make_model(kind, int(y.sum()), int((y == 0).sum()))
            model.fit(train[FEATURE_SETS[feature_set]], y)
            scores[test] = model.predict_proba(table.loc[test, FEATURE_SETS[feature_set]])[:, 1]
        out[key] = scores
    return out


def recall_at(predictions: pd.DataFrame, key: str, share: float) -> tuple[int, int]:
    """Landslide cells inside the top `share` of each fold's land, summed over folds."""
    hits = total = 0
    for _, part in predictions.groupby("fold"):
        cut = part[key].quantile(1 - share)
        positive = part["landslide"] == 1
        hits += int((positive & (part[key] >= cut)).sum())
        total += int(positive.sum())
    return hits, total


def block_bootstrap(predictions: pd.DataFrame, key: str, reps: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    groups = {b: g for b, g in predictions.groupby("block")}
    names = list(groups)
    values = []
    for _ in range(reps):
        sample = pd.concat([groups[b] for b in rng.choice(names, size=len(names))])
        if sample["landslide"].sum() > 0:
            values.append(average_precision_score(sample["landslide"], sample[key]))
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, (_, _, label_text) in METHODS.items():
        low, high = block_bootstrap(predictions, key)
        row = {
            "key": key,
            "method": label_text,
            "pr_auc": average_precision_score(predictions["landslide"], predictions[key]),
            "pr_auc_low": low,
            "pr_auc_high": high,
        }
        for share in TOP_SHARES:
            hits, total = recall_at(predictions, key, share)
            row[f"recall_{int(share * 100)}"] = hits / total if total else np.nan
            row[f"hits_{int(share * 100)}"] = hits
        rows.append(row)
    return pd.DataFrame(rows)


def train_final(table: pd.DataFrame) -> tuple[XGBClassifier, dict]:
    features = FEATURE_SETS[FINAL]
    y = table["landslide"]
    model = make_model("xgboost", int(y.sum()), int((y == 0).sum()))
    model.fit(table[features], y)
    meta = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": features,
        "region": REGION,
        "cell_deg": CELL,
        "positives": int(y.sum()),
        "cells": len(table),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return model, meta


def save_raster(table: pd.DataFrame, values: np.ndarray) -> None:
    shape = grid_shape()
    grid = np.full(shape, np.nan, dtype="float32")
    grid[table["row"], table["col"]] = values
    west, _, _, north = REGION
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "crs": "EPSG:4326",
        "nodata": np.nan,
        "height": shape[0],
        "width": shape[1],
        "transform": from_origin(west, north, CELL, CELL),
        "compress": "deflate",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(RASTER_PATH, "w", **profile) as dst:
        dst.write(grid, 1)


def held_out_percentile(predictions: pd.DataFrame, key: str) -> np.ndarray:
    """Each cell's held-out score as a percentile of the land in its own fold."""
    return predictions.groupby("fold")[key].rank(pct=True).to_numpy()


def ward_summary(table: pd.DataFrame, percentile: np.ndarray) -> pd.DataFrame:
    """Mumbai wards: the cells whose centre falls inside each ward, and the highest percentile."""
    wards = static.load_wards()
    points = gpd.GeoDataFrame(
        {"percentile": percentile, "landslide": table["landslide"].to_numpy()},
        geometry=gpd.points_from_xy(table["lon"], table["lat"]),
        crs=4326,
    )
    joined = gpd.sjoin(points, wards, predicate="within")
    out = joined.groupby("ward_id").agg(
        cells=("percentile", "size"),
        reported=("landslide", "sum"),
        top_percentile=("percentile", "max"),
    )
    return out.sort_values("top_percentile", ascending=False).round(3).reset_index()


# --- Report ---------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _row(cells_: list) -> str:
    return "| " + " | ".join(str(c) for c in cells_) + " |"


def report(table, inventory, scored, importance, wards, command) -> str:
    positives = int(table["landslide"].sum())
    prevalence = positives / len(table)
    good = inventory["accuracy"].isin(GOOD_ACCURACY)
    by_source = inventory[good].groupby("source").size().to_dict()
    final = scored.set_index("key").loc["xgb_terrain"]
    slope = scored.set_index("key").loc["slope"]
    cover = scored.set_index("key").loc["xgb_cover"]
    roads = scored.set_index("key").loc["xgb_roads"]
    sites = table[table["landslide"] == 1]
    first, last = (
        min(d for d in inventory["date"] if pd.notna(d)),
        max(d for d in inventory["date"] if pd.notna(d)),
    )
    lines = [
        "# Landslide susceptibility: MMR and Raigad",
        "",
        f"Generated by `{command}`. Do not edit by hand.",
        "",
        "## Result",
        "",
        f"- **Held-out areas, final model (XGBoost on terrain and rain):** PR-AUC "
        f"{final['pr_auc']:.3f} (90% range "
        f"{final['pr_auc_low']:.3f} to {final['pr_auc_high']:.3f}) against {prevalence:.4f} for "
        f"a random guess. The top 10% most susceptible land of each held-out area held "
        f"{final['hits_10']} of {positives} landslide cells (recall {final['recall_10']:.0%}); "
        f"the top 20% held {final['hits_20']} ({final['recall_20']:.0%}).",
        f"- **Slope alone:** PR-AUC {slope['pr_auc']:.3f}, recall {slope['recall_10']:.0%} in "
        f"the top 10% and {slope['recall_20']:.0%} in the top 20%.",
        f"- **The inventory is mostly urban.** The median landslide cell is "
        f"{sites['built_frac'].median():.0%} built-up and {sites['elev_mean_m'].median():.0f} m "
        f"high; the median land cell is {table['built_frac'].median():.0%} built-up and "
        f"{table['elev_mean_m'].median():.0f} m high. News reports landslides on hillside "
        "settlements in Mumbai and Thane, on the expressway and on railways, and rarely those in "
        "the forested Ghats. That is why slope alone ranks them poorly.",
        f"- **Adding land cover** gives PR-AUC {cover['pr_auc']:.3f} (recall "
        f"{cover['recall_10']:.0%} in the top 10%), and **also road distance** "
        f"{roads['pr_auc']:.3f} ({roads['recall_10']:.0%}). Those scores are higher because the "
        "model learns where landslides get reported: with land cover, built-up and low "
        "vegetation shares were its two strongest features and nearly every Mumbai ward, flat "
        "ones included, came out in the top 1% of the area. Both are left out of the final model "
        "(land cover after that first run; see Setup).",
        f"- **Few landslides:** {positives} landslide cells. Moving two or three of them between "
        "areas changes these scores a lot; the ranges above show how much.",
        "",
        "## Setup",
        "",
        f"- **Area:** {REGION[0]}-{REGION[2]} E, {REGION[1]}-{REGION[3]} N, a box around the "
        "Mumbai Metropolitan Region and Raigad district (it also takes in the Pune district "
        f"ghats at Lonavala and Khandala). {len(table)} land cells of 0.01 degree (about "
        "1.1 x 1.05 km).",
        f"- **Inventory:** NASA Global Landslide Catalog, events {first} to {last} "
        f"({len(inventory)} in the area, {int(good.sum())} located to 1 km or better and used: "
        f"{by_source}). It is built from news reports, so it over-represents landslides that hit "
        "roads, railways and homes and misses those in forest. It ends in 2017: Taliye "
        "(Mahad, 2021) and Irshalwadi (Khalapur, 2023) are not in it.",
        "- **GSI inventory:** the Geological Survey of India's Bhukosh portal needs a login and "
        "did not respond from this machine. Its export placed in `data/raw/gsi/` is used "
        "automatically on the next run.",
        "- **Order of decisions:** road distance was left out before any score was seen. Land "
        "cover was left out after the first run, on the feature importance and the ward list, "
        "not on the scores; that first run is the 'plus land cover' row below.",
        "- **Cells without a reported landslide count as no landslide.** Many of them are "
        "unreported landslides, so precision here is a lower bound.",
        "- **Features:** "
        + ", ".join(f"`{f}`" for f in FEATURE_SETS["with_roads"])
        + ". Terrain from Copernicus DEM (30 m; a surface model, so buildings and trees add "
        "height), land cover from ESA WorldCover 2021 read at 40 m, distances from "
        "OpenStreetMap, June-September rain from IMD (0.25 degree; the years on disk).",
        f"- **Split by area:** {BLOCK} degree blocks (about 27 km), {FOLDS} folds, each block held "
        "out once. Scores pool every held-out prediction.",
        "- **Recall at the top 10% / 20%:** within each held-out fold, the share of its "
        "landslide cells that fall in its most susceptible 10% / 20% of land.",
        "",
        "## Held-out scores",
        "",
        _row(["Method", "PR-AUC", "90% range", "Recall, top 10%", "Recall, top 20%"]),
        _row(["---"] * 5),
    ]
    for _, r in scored.iterrows():
        lines.append(
            _row(
                [
                    r["method"],
                    f"{r['pr_auc']:.3f}",
                    f"{r['pr_auc_low']:.3f} to {r['pr_auc_high']:.3f}",
                    f"{r['hits_10']} of {positives} ({r['recall_10']:.0%})",
                    f"{r['hits_20']} of {positives} ({r['recall_20']:.0%})",
                ]
            )
        )
    lines += [
        "",
        f"A random ranking scores PR-AUC {prevalence:.4f} and recall 10% / 20%.",
        "",
        "## What the final model leans on",
        "",
        _row(["Feature", "Share of gain"]),
        _row(["---"] * 2),
        *[_row([f, f"{s:.2f}"]) for f, s in importance.items()],
        "",
        "## Mumbai wards",
        "",
        "For each ward, its most susceptible cell as a percentile of the land in its "
        "held-out fold (1.00 = the most susceptible), scored by the final model's held-out run, "
        "which never saw that area. 'Reported' counts the ward's cells with a catalog "
        "landslide. Cells are 1 km, so a ward's hill slopes share a cell with flat ground.",
        "",
        _row(["Ward", "Cells", "Reported", "Most susceptible cell (percentile)"]),
        _row(["---"] * 4),
        *[
            _row([r["ward_id"], r["cells"], int(r["reported"]), f"{r['top_percentile']:.2f}"])
            for _, r in wards.iterrows()
        ],
        "",
        "Rain and elevation are also partly a location: rain rises steadily from the coast to "
        "the Ghats, and the reported landslides cluster in Mumbai and Thane. So even without "
        "land cover the model partly learns where reports come from. Only a field inventory "
        "(GSI) can separate the two.",
        "",
        "## Limits",
        "",
        "- No geology: the GSI lithology and structure maps are also on Bhukosh.",
        "- Susceptibility only (where). When a landslide happens depends on rain over the "
        "previous days; a rain trigger on top of this map is later work.",
        "- 1 km cells are coarse for landslides, which are tens to hundreds of metres across; "
        "they match the location accuracy of the inventory, not the size of the slopes.",
        "",
        "## Next",
        "",
        "- Get the GSI inventory (Bhukosh login) and rerun; add Taliye and Irshalwadi.",
        "- Add lithology; then a rain trigger using the Stage 4 rain forecast.",
        "",
        f"Inventory citation: Kirschbaum, D. B. et al. (2010), A global landslide catalog for "
        f"hazard applications, Natural Hazards 52; NASA Global Landslide Catalog export, "
        f"accessed {_today()}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Landslide susceptibility (Stage 10).")
    parser.add_argument("--fetch", action="store_true", help="download what is missing")
    parser.add_argument("--rebuild", action="store_true", help="recompute the cell features")
    parser.add_argument("--report", type=Path, help="score, save the final model, write report")
    args = parser.parse_args(argv)
    if not (args.fetch or args.report):
        parser.error("give --fetch and/or --report")
    if args.fetch:
        fetch()
    if args.report:
        inventory = events()
        table = label(cells(rebuild=args.rebuild), inventory)
        print(f"{len(table)} land cells, {int(table['landslide'].sum())} with a landslide")
        predictions = held_out(table)
        scored = score(predictions)
        print(scored.drop(columns="key").round(3).to_string(index=False))
        model, meta = train_final(table)
        values = model.predict_proba(table[meta["features"]])[:, 1]
        save_raster(table, values)
        wards = ward_summary(table, held_out_percentile(predictions, "xgb_terrain"))
        wards.to_csv(WARDS_PATH, index=False)
        gain = model.get_booster().get_score(importance_type="total_gain")
        importance = pd.Series({f: gain.get(f, 0.0) for f in meta["features"]})
        importance = (importance / importance.sum()).sort_values(ascending=False)
        command = "python -m models.landslide " + " ".join(
            argv if argv is not None else sys.argv[1:]
        )
        args.report.write_text(
            report(table, inventory, scored, importance, wards, command), encoding="utf-8"
        )
        print(f"Saved {MODEL_PATH}, {RASTER_PATH}; wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
