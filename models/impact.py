"""Vulnerability, impact and routes (Stage 12): who lives where, how far each ward is from a
shelter and a hospital by road, how many people a risk run puts at risk, and a route from each
ward to its nearest shelter and hospital that keeps off roads likely to be under water.

Population  Meta and CIESIN High Resolution Settlement Layer (HRSL, general population), 1
            arc-second cells (about 30 m), read as a window from the public cloud-optimised
            files on AWS (CC BY 4.0). Census counts spread over buildings found in satellite
            images: an estimate of where people live, not a count.
Facilities  OpenStreetMap through the Overpass API. Hospitals: amenity=hospital (many are small
            nursing homes; emergency=yes is kept as a flag). Shelter candidates: schools,
            community centres, emergency assembly points and social-facility shelters. BMC opens
            municipal schools as temporary shelters during floods, but its list is not open
            data, so these are candidates, not designated shelters. OSM amenity=shelter is left
            out: in Mumbai it is mostly park gazebos and bus stops. An official list (CSV with
            name, lat, lon) put in data/raw/shelters/ replaces the OSM candidates.
Roads       OSMnx drive network inside the wards plus 1 km, treated as undirected (people walk
            both ways down one-way streets). Footpaths are not in it, so walking distances are
            a little long. Facilities more than 300 m from any junction are dropped.
Distances   Every populated cell joins its nearest junction. One search from all shelters (and
            one from all hospitals) gives the road distance from every junction to the nearest
            one. A cell's distance = straight line to its junction + road distance + straight
            line from the facility's junction.
Vulnerability  Four indicators per ward, each turned into its rank among the 24 wards (0 lowest,
            1 highest) and averaged with equal weights: population density, share of people on
            low-lying land (below 5 m in the Copernicus DEM, a surface model), mean road
            distance to the nearest shelter candidate, and to the nearest hospital. The index
            orders wards; it says nothing about absolute need. No open data per ward on income,
            age, housing or slums was found.
Impact      For a risk run: people at risk = probability x people exposed (flood: people on
            low-lying land; heat: everyone in the ward). Impact score = people at risk x
            (0.5 + vulnerability), so the most vulnerable ward weighs up to three times the
            least.
Routes      In a flood run, a road segment is avoided when its ward is orange or red and the
            segment's midpoint is on low-lying land or within 250 m of a reported waterlogging
            spot (flood labels, medium confidence). Routes start at the junction nearest each
            ward's population-weighted centre and end at the nearest shelter candidate or
            hospital reachable without avoided segments. Heat runs avoid nothing. "baseline"
            is the run id for routes that avoid nothing; the API falls back to it.

python -m models.impact --fetch                               population, facilities, roads
python -m models.impact --build --report docs/impact.md       vulnerability per ward -> PostGIS
python -m models.impact --routes RUN_ID                       routes for a run (or 'baseline')
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from sqlalchemy import delete, select, text, update

from api.config import ROOT, settings
from api.db import Facility, SessionLocal, Ward, WardRiskRow, WardRoute, WardVulnerability, init_db
from api.schemas import Hazard, RouteStatus, Severity
from pipelines import static, store

HRSL_URL = (
    "https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/hrsl_general/hrsl_general-latest.vrt"
)
ROAD_BUFFER_M = 1000  # road network beyond the ward edge, so routes may leave a ward
MAX_SNAP_M = 300  # facilities further than this from any junction are outside the network
LOW_LYING_M = static.LOW_LYING_M
SPOT_RADIUS_M = 250
AVOID_LEVELS = (Severity.ORANGE, Severity.RED)
SHELTER_FAR_KM = 1.0
HOSPITAL_FAR_KM = 2.0
INDICATORS = {  # column: what a higher value means
    "density_per_km2": "more people per km2",
    "low_lying_share": "more people on low-lying land",
    "shelter_km": "further from a shelter candidate",
    "hospital_km": "further from a hospital",
}
SHELTER_KINDS = {
    "school": "school",
    "community_centre": "community centre",
    "assembly_point": "assembly point",
    "shelter": "shelter",
}
SPECIALIST = (  # names of places tagged amenity=hospital that do not take general emergencies
    r"\b(?:eye|vision|optic(?:al|s)?|spectacles?|lens|laser|dental|dentist|teeth|diagnostics?|"
    r"patholog\w*|labs?|laborator\w*|imaging|scans?|x-?ray|ivf|(?:in)?fertility|skin|derma\w*|"
    r"cosmetic|hair|physio\w*|veterinary|pet|animal|research|blood bank|ayurved\w*|homo?eo\w*)\b"
)
GENERAL = (
    r"\b(?:multi-?special\w*|nursing home|general|surgical|maternity|trauma|critical|criticare)\b"
)
BASELINE = "baseline"
KINDS = ("shelter", "hospital")
TO_METRIC = Transformer.from_crs("EPSG:4326", static.METRIC_CRS, always_xy=True)


def population_path() -> Path:
    return settings.raw_dir / "population" / "hrsl_mumbai.tif"


def facilities_path() -> Path:
    return settings.raw_dir / "osm" / "facilities.json"


def roads_path() -> Path:
    return settings.raw_dir / "osm" / "drive.graphml"


def official_shelters_dir() -> Path:
    return settings.raw_dir / "shelters"


def vulnerability_path() -> Path:
    return settings.processed_dir / "impact" / "ward_vulnerability.csv"


# --- Download -------------------------------------------------------------------------------


def _area(wards: gpd.GeoDataFrame):
    """The wards plus ROAD_BUFFER_M, as one WGS84 polygon."""
    metric = wards.to_crs(static.METRIC_CRS).buffer(ROAD_BUFFER_M).union_all()
    return gpd.GeoSeries([metric], crs=static.METRIC_CRS).to_crs(4326).iloc[0]


def fetch_population(wards: gpd.GeoDataFrame) -> None:
    path = population_path()
    if path.exists():
        return
    bounds = static._bounds(wards)
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4")
    with env, rasterio.open(f"/vsicurl/{HRSL_URL}") as src:
        window = from_bounds(*bounds, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=window).astype("float32")
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "crs": src.crs,
            "nodata": np.nan,
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": window_transform(window, src.transform),
            "compress": "deflate",
        }
    static._save(path, data, profile, HRSL_URL)


def fetch_facilities(wards: gpd.GeoDataFrame) -> None:
    path = facilities_path()
    if path.exists():
        return
    west, south, east, north = static._bounds(wards)
    box = f"({south},{west},{north},{east})"
    query = (
        "[out:json][timeout:180];("
        f'nwr["amenity"="hospital"]{box};'
        f'nwr["amenity"="school"]{box};'
        f'nwr["amenity"="community_centre"]{box};'
        f'nwr["emergency"="assembly_point"]{box};'
        f'nwr["social_facility"="shelter"]{box};'
        ");out tags center;"
    )
    resp = store.http_get(static.OVERPASS_URL, params={"data": query}, timeout=240)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    store.log_download("osm", datetime.now(UTC).date(), path, static.OVERPASS_URL)


def fetch_roads(wards: gpd.GeoDataFrame) -> None:
    path = roads_path()
    if path.exists():
        return
    import osmnx as ox

    ox.settings.http_user_agent = store.USER_AGENT
    ox.settings.use_cache = False
    ox.settings.requests_timeout = 300
    for attempt in range(3):  # the public Overpass server drops connections when busy
        try:
            graph = ox.graph_from_polygon(_area(wards), network_type="drive", simplify=True)
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(60)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    ox.save_graphml(graph, tmp)
    tmp.replace(path)
    store.log_download("osm", datetime.now(UTC).date(), path, ox.settings.overpass_url)


def fetch() -> None:
    wards = static.load_wards()
    for name, step in (
        ("population", fetch_population),
        ("facilities", fetch_facilities),
        ("roads", fetch_roads),
    ):
        print(f"  {name} ...", flush=True)
        step(wards)


# --- Inputs ---------------------------------------------------------------------------------


def inputs_ready() -> bool:
    """True once --fetch has run (routes need population, facilities and roads)."""
    paths = (population_path(), facilities_path(), roads_path(), static.dem_path())
    return all(p.exists() for p in paths)


def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m models.impact --fetch")
    return path


@lru_cache(maxsize=1)
def _dem() -> tuple[np.ndarray, rasterio.Affine]:
    path = static.dem_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m pipelines.static")
    with rasterio.open(path) as src:
        return src.read(1).astype("float64"), src.transform


def elevation_at(lon, lat) -> np.ndarray:
    """Copernicus DEM height (m) at each point; NaN outside the DEM."""
    heights, transform = _dem()
    lon, lat = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    col = np.floor((lon - transform.c) / transform.a).astype(int)
    row = np.floor((lat - transform.f) / transform.e).astype(int)
    inside = (row >= 0) & (row < heights.shape[0]) & (col >= 0) & (col < heights.shape[1])
    out = np.full(lon.shape, np.nan)
    out[inside] = heights[row[inside], col[inside]]
    return out


def population_cells(wards: gpd.GeoDataFrame) -> pd.DataFrame:
    """Populated cells inside the wards: lon, lat, x, y (metres), people, ward_id, low_lying."""
    with rasterio.open(_need(population_path())) as src:
        people = src.read(1)
        transform = src.transform
    labels = static.ward_labels(wards, people.shape, transform)
    rows, cols = np.nonzero((labels > 0) & np.isfinite(people) & (people > 0))
    lon, lat = rasterio.transform.xy(transform, rows, cols)
    lon, lat = np.asarray(lon), np.asarray(lat)
    x, y = TO_METRIC.transform(lon, lat)
    elevation = elevation_at(lon, lat)
    return pd.DataFrame(
        {
            "lon": lon,
            "lat": lat,
            "x": x,
            "y": y,
            "people": people[rows, cols].astype("float64"),
            "ward_id": wards["ward_id"].to_numpy()[labels[rows, cols] - 1],
            "low_lying": elevation < LOW_LYING_M,
        }
    )


def _official_shelters() -> pd.DataFrame | None:
    files = (
        sorted(official_shelters_dir().glob("*.csv")) if official_shelters_dir().exists() else []
    )
    if not files:
        return None
    table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    table.columns = [c.strip().lower() for c in table.columns]
    missing = {"name", "lat", "lon"} - set(table.columns)
    if missing:
        raise ValueError(f"official shelter files need columns name, lat, lon; missing {missing}")
    return pd.DataFrame(
        {
            "facility_id": [f"official-{i + 1}" for i in range(len(table))],
            "kind": "shelter",
            "name": table["name"].astype(str),
            "lat": table["lat"].astype(float),
            "lon": table["lon"].astype(float),
            "emergency": False,
            "official": True,
        }
    )


def parse_facilities(elements: list[dict]) -> pd.DataFrame:
    """Overpass elements (out tags center) to one row per facility."""
    rows = []
    for e in elements:
        tags = e.get("tags", {})
        lat = e.get("lat", e.get("center", {}).get("lat"))
        lon = e.get("lon", e.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue
        if tags.get("amenity") == "hospital":
            kind = "hospital"
        elif tags.get("amenity") in ("school", "community_centre"):
            kind = tags["amenity"]
        elif tags.get("emergency") == "assembly_point":
            kind = "assembly_point"
        elif tags.get("social_facility") == "shelter":
            kind = "shelter"
        else:
            continue
        rows.append(
            {
                "facility_id": f"osm-{e['type']}-{e['id']}",
                "kind": kind,
                "name": tags.get("name:en") or tags.get("name"),
                "lat": float(lat),
                "lon": float(lon),
                "emergency": tags.get("emergency") == "yes",
                "official": False,
            }
        )
    columns = ["facility_id", "kind", "name", "lat", "lon", "emergency", "official"]
    return pd.DataFrame(rows, columns=columns).drop_duplicates("facility_id")


def load_facilities(wards: gpd.GeoDataFrame) -> pd.DataFrame:
    """Hospitals and shelter candidates with their ward (None outside the wards)."""
    elements = json.loads(_need(facilities_path()).read_text(encoding="utf-8"))["elements"]
    table = parse_facilities(elements)
    keep = usable(table)
    dropped = table[~keep]
    table = table[keep].reset_index(drop=True)
    counts = {
        "unnamed": int(dropped["name"].isna().sum()),
        "specialist": int((dropped["name"].notna() & (dropped["kind"] == "hospital")).sum()),
    }
    official = _official_shelters()
    if official is not None:
        table = pd.concat([table[table["kind"] == "hospital"], official], ignore_index=True)
    points = gpd.GeoDataFrame(
        table, geometry=gpd.points_from_xy(table["lon"], table["lat"]), crs=4326
    )
    joined = gpd.sjoin(points, wards, how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]
    table["ward_id"] = _ward_ids(joined["ward_id"])
    table.attrs["dropped"] = counts
    return table


def specialist(name: str) -> bool:
    """True for a name like "Krishna Eye Care Centre" or "SRL Diagnostics", but not for a
    general hospital that mentions a speciality ("Bombay Hospital and Medical Research Center",
    "Maternity Surgical and Diagnostic Centre")."""
    found = re.search(SPECIALIST, name, re.IGNORECASE)
    if not found or re.search(GENERAL, name, re.IGNORECASE):
        return False
    hospital = re.search(r"\bhospital\b", name, re.IGNORECASE)
    return not (hospital and hospital.start() < found.start())


def usable(table: pd.DataFrame) -> pd.Series:
    """Facilities a route can point people to: named, and for hospitals not a specialist
    clinic or lab (eye, dental, diagnostic and the like are often tagged amenity=hospital)."""
    names = table["name"].fillna("").astype(str).str.strip()
    is_specialist = names.map(specialist).astype(bool)
    return (names.str.len() > 0) & ~((table["kind"] == "hospital") & is_specialist)


def _ward_ids(values) -> list[str | None]:
    """Ward ids with None (not NaN) outside the wards, ready for the database."""
    return [w if isinstance(w, str) else None for w in values]


def waterlogging_spots() -> pd.DataFrame:
    """Distinct places reported waterlogged in the flood labels (medium confidence or better)."""
    labels = pd.read_csv(ROOT / "labels" / "flood_events.csv")
    spots = labels[
        (labels["flooded"] == 1)
        & labels["confidence"].isin(["medium", "high"])
        & labels["lat"].notna()
    ]
    return spots.drop_duplicates(["lat", "lon"])[["location", "lat", "lon"]].reset_index(drop=True)


# --- Road network ---------------------------------------------------------------------------


@dataclass
class Roads:
    """Undirected road graph with junction coordinates and per-segment flags."""

    graph: nx.Graph
    nodes: np.ndarray  # node ids, in the order of xy
    xy: np.ndarray  # (n, 2) metres
    lonlat: np.ndarray  # (n, 2)
    edges: pd.DataFrame  # u, v, length, ward_id, low_lying, near_spot
    tree: cKDTree

    @classmethod
    def from_graph(cls, graph: nx.Graph, wards=None, spots=None) -> "Roads":
        nodes = np.array(list(graph.nodes))
        lonlat = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in nodes], dtype=float)
        xy = np.column_stack(TO_METRIC.transform(lonlat[:, 0], lonlat[:, 1]))
        edges = pd.DataFrame(
            [(u, v, d["length"], d.get("geometry")) for u, v, d in graph.edges(data=True)],
            columns=["u", "v", "length", "geometry"],
        )
        mids = [
            geom.interpolate(0.5, normalized=True)
            if geom is not None
            else Point(
                (graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2,
                (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2,
            )
            for u, v, geom in zip(edges["u"], edges["v"], edges["geometry"], strict=True)
        ]
        mid_lon = np.array([p.x for p in mids])
        mid_lat = np.array([p.y for p in mids])
        edges["low_lying"] = elevation_at(mid_lon, mid_lat) < LOW_LYING_M
        edges["near_spot"] = False
        if spots is not None and len(spots):
            sx, sy = TO_METRIC.transform(spots["lon"].to_numpy(), spots["lat"].to_numpy())
            mx, my = TO_METRIC.transform(mid_lon, mid_lat)
            near = cKDTree(np.column_stack([sx, sy])).query(np.column_stack([mx, my]))[0]
            edges["near_spot"] = near <= SPOT_RADIUS_M
        edges["ward_id"] = None
        if wards is not None:
            points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(mid_lon, mid_lat), crs=4326)
            joined = gpd.sjoin(points, wards, how="left", predicate="within")
            joined = joined[~joined.index.duplicated()]
            edges["ward_id"] = _ward_ids(joined["ward_id"])
        return cls(graph, nodes, xy, lonlat, edges.drop(columns="geometry"), cKDTree(xy))

    def nearest(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """(node id, straight-line metres) of the junction nearest each metric point."""
        dist, idx = self.tree.query(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))
        return self.nodes[idx], dist

    def avoided(self, levels: dict[str, Severity], hazard: Hazard) -> pd.DataFrame:
        """Segments to keep off: in an orange or red ward, low-lying or near a waterlogging spot."""
        if hazard != Hazard.FLOOD or not levels:
            return self.edges.iloc[0:0]
        bad = {w for w, level in levels.items() if level in AVOID_LEVELS}
        e = self.edges
        return e[e["ward_id"].isin(bad) & (e["low_lying"] | e["near_spot"])]


def load_roads(wards: gpd.GeoDataFrame | None = None) -> Roads:
    import osmnx as ox

    multi = ox.load_graphml(_need(roads_path()))
    graph = nx.Graph()
    for n, d in multi.nodes(data=True):
        graph.add_node(n, x=d["x"], y=d["y"])
    for u, v, d in multi.edges(data=True):  # keep the shortest of parallel and two-way edges
        if u == v:
            continue
        if not graph.has_edge(u, v) or d["length"] < graph[u][v]["length"]:
            graph.add_edge(u, v, length=float(d["length"]), geometry=d.get("geometry"))
    wards = static.load_wards() if wards is None else wards
    return Roads.from_graph(graph, wards, waterlogging_spots())


@dataclass
class Reach:
    """Road distance from every junction to its nearest facility, and the path there."""

    distance: dict  # node -> metres, including the facility's own snap distance
    facility: dict  # node -> facility_id
    paths: dict  # node -> [node, ..., facility node]

    def to(self, node) -> float:
        return self.distance.get(node, np.inf)


SOURCE = "__facilities__"


def reach(roads: Roads, facilities: pd.DataFrame, avoid: pd.DataFrame | None = None) -> Reach:
    """One Dijkstra search out from all facilities at once (from a virtual node joined to each
    facility's junction by its snap distance), keeping off the `avoid` segments."""
    fx, fy = TO_METRIC.transform(facilities["lon"].to_numpy(), facilities["lat"].to_numpy())
    nodes, snap = roads.nearest(fx, fy)
    best: dict = {}
    for node, d, fid in zip(nodes, snap, facilities["facility_id"], strict=True):
        if d <= MAX_SNAP_M and (node not in best or d < best[node][0]):
            best[node] = (float(d), fid)
    graph = roads.graph
    if avoid is not None and len(avoid):
        graph = nx.restricted_view(graph, [], list(zip(avoid["u"], avoid["v"], strict=True)))
    graph = nx.Graph(graph)  # a copy, so the virtual node never touches the shared graph
    for node, (d, _) in best.items():
        graph.add_edge(SOURCE, node, length=d)
    if not best:
        return Reach({}, {}, {})
    distance, paths = nx.single_source_dijkstra(graph, SOURCE, weight="length")
    distance.pop(SOURCE, None)
    paths.pop(SOURCE, None)
    paths = {n: list(reversed(p[1:])) for n, p in paths.items()}
    facility = {n: best[p[-1]][1] for n, p in paths.items()}
    return Reach(distance, facility, paths)


def route_line(roads: Roads, path: list) -> LineString | None:
    """The road geometry along a path of junctions, end to end."""
    if len(path) < 2:
        return None
    coords: list[tuple[float, float]] = []
    for a, b in zip(path[:-1], path[1:], strict=True):
        geom = roads.graph[a][b].get("geometry")
        a_xy = (roads.graph.nodes[a]["x"], roads.graph.nodes[a]["y"])
        b_xy = (roads.graph.nodes[b]["x"], roads.graph.nodes[b]["y"])
        part = list(geom.coords) if geom is not None else [a_xy, b_xy]
        if np.hypot(part[0][0] - a_xy[0], part[0][1] - a_xy[1]) > np.hypot(
            part[-1][0] - a_xy[0], part[-1][1] - a_xy[1]
        ):
            part.reverse()
        coords.extend(part if not coords else part[1:])
    return LineString(coords)


# --- Vulnerability per ward -----------------------------------------------------------------


def ranked(values: pd.Series) -> pd.Series:
    """Rank among the wards scaled to 0 (lowest) .. 1 (highest); ties share the mean rank."""
    if len(values) < 2:
        return pd.Series(0.5, index=values.index)
    return (values.rank(method="average") - 1) / (len(values) - 1)


def vulnerability_index(table: pd.DataFrame) -> pd.Series:
    return pd.concat([ranked(table[c]) for c in INDICATORS], axis=1).mean(axis=1).round(3)


def cell_distances(roads: Roads, cells: pd.DataFrame, found: Reach) -> np.ndarray:
    """Metres from each cell to its nearest facility by road (inf when none is reachable)."""
    nodes, snap = roads.nearest(cells["x"].to_numpy(), cells["y"].to_numpy())
    return snap + np.array([found.to(n) for n in nodes])


def ward_table(
    wards: gpd.GeoDataFrame, cells: pd.DataFrame, shelter_m: np.ndarray, hospital_m: np.ndarray
) -> pd.DataFrame:
    cells = cells.assign(shelter_m=shelter_m, hospital_m=hospital_m)
    area = wards.set_index("ward_id").to_crs(static.METRIC_CRS).area / 1e6
    out = []
    for ward_id, group in cells.groupby("ward_id"):
        w = group["people"]
        total = float(w.sum())

        def mean_km(column, group=group, w=w):
            ok = np.isfinite(group[column])
            return float(np.average(group.loc[ok, column], weights=w[ok]) / 1000)

        out.append(
            {
                "ward_id": ward_id,
                "population": int(round(total)),
                "area_km2": round(float(area[ward_id]), 2),
                "density_per_km2": round(total / float(area[ward_id])),
                "low_lying_pop": int(round(float(w[group["low_lying"]].sum()))),
                "low_lying_share": round(float(w[group["low_lying"]].sum()) / total, 3),
                "shelter_km": round(mean_km("shelter_m"), 2),
                "hospital_km": round(mean_km("hospital_m"), 2),
                "shelter_far_share": round(
                    float(w[group["shelter_m"] > SHELTER_FAR_KM * 1000].sum()) / total, 3
                ),
                "hospital_far_share": round(
                    float(w[group["hospital_m"] > HOSPITAL_FAR_KM * 1000].sum()) / total, 3
                ),
            }
        )
    table = pd.DataFrame(out).set_index("ward_id")
    table["vulnerability_index"] = vulnerability_index(table)
    return table


def origins(roads: Roads, cells: pd.DataFrame, wards: gpd.GeoDataFrame) -> pd.DataFrame:
    """Start per ward: the junction inside the ward nearest its population-weighted centre (any
    junction if none is inside), and the straight-line metres from the centre to it."""
    joined = gpd.sjoin(
        gpd.GeoDataFrame(
            {"node": roads.nodes},
            geometry=gpd.points_from_xy(roads.lonlat[:, 0], roads.lonlat[:, 1]),
            crs=4326,
        ),
        wards,
        how="left",
        predicate="within",
    )
    node_ward = joined[~joined.index.duplicated()]["ward_id"].to_numpy()
    out = {}
    for ward_id, group in cells.groupby("ward_id"):
        cx = np.average(group["x"], weights=group["people"])
        cy = np.average(group["y"], weights=group["people"])
        inside = np.flatnonzero(node_ward == ward_id)
        if not len(inside):
            inside = np.arange(len(roads.nodes))
        d = np.hypot(roads.xy[inside, 0] - cx, roads.xy[inside, 1] - cy)
        best = int(np.argmin(d))
        out[ward_id] = {"node": roads.nodes[inside[best]], "snap": float(d[best])}
    return pd.DataFrame.from_dict(out, orient="index")


@dataclass
class Inputs:
    wards: gpd.GeoDataFrame
    cells: pd.DataFrame
    facilities: pd.DataFrame
    roads: Roads

    @classmethod
    def load(cls) -> "Inputs":
        wards = static.load_wards()
        print("  population cells ...", flush=True)
        cells = population_cells(wards)
        facilities = load_facilities(wards)
        print("  road graph ...", flush=True)
        roads = load_roads(wards)
        return cls(wards, cells, facilities, roads)

    def of_kind(self, kind: str) -> pd.DataFrame:
        f = self.facilities
        return f[f["kind"] == "hospital"] if kind == "hospital" else f[f["kind"] != "hospital"]


def build(inputs: Inputs) -> pd.DataFrame:
    shelters = reach(inputs.roads, inputs.of_kind("shelter"))
    hospitals = reach(inputs.roads, inputs.of_kind("hospital"))
    table = ward_table(
        inputs.wards,
        inputs.cells,
        cell_distances(inputs.roads, inputs.cells, shelters),
        cell_distances(inputs.roads, inputs.cells, hospitals),
    )
    path = vulnerability_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    return table


def save_vulnerability(table: pd.DataFrame, facilities: pd.DataFrame) -> None:
    init_db()
    now = datetime.now(UTC)
    with SessionLocal.begin() as session:
        session.execute(delete(WardVulnerability))
        for ward_id, row in table.iterrows():
            session.add(WardVulnerability(ward_id=ward_id, built_at=now, **row.to_dict()))
            session.execute(
                update(Ward)
                .where(Ward.ward_id == ward_id)
                .values(
                    population=int(row["population"]),
                    vulnerability_index=float(row["vulnerability_index"]),
                )
            )
        session.execute(delete(Facility))
        for f in facilities.itertuples():
            session.add(
                Facility(
                    facility_id=f.facility_id,
                    kind=f.kind,
                    name=(f.name or None) if isinstance(f.name, str) else None,
                    ward_id=f.ward_id if isinstance(f.ward_id, str) else None,
                    emergency=bool(f.emergency),
                    official=bool(f.official),
                    geometry=f"SRID=4326;POINT({f.lon} {f.lat})",
                )
            )


# --- Routes for a run -----------------------------------------------------------------------


def run_levels(run_id: str) -> tuple[Hazard | None, dict[str, Severity]]:
    if run_id == BASELINE:
        return None, {}
    with SessionLocal() as session:
        rows = session.execute(
            select(WardRiskRow.ward_id, WardRiskRow.hazard, WardRiskRow.severity).where(
                WardRiskRow.run_id == run_id
            )
        ).all()
    if not rows:
        raise RuntimeError(f"no ward rows for run {run_id}; see python -m models.risk --list")
    return Hazard(rows[0].hazard), {r.ward_id: Severity(r.severity) for r in rows}


def routes(inputs: Inputs, run_id: str) -> list[dict]:
    """Shelter and hospital route per ward for a run, keeping off the avoided segments."""
    hazard, levels = run_levels(run_id)
    roads = inputs.roads
    avoid = roads.avoided(levels, hazard)
    start = origins(roads, inputs.cells, inputs.wards)
    names = inputs.facilities.set_index("facility_id")["name"]
    avoided_km = (avoid.groupby("ward_id")["length"].sum() / 1000).round(2)
    cell_nodes, _ = roads.nearest(inputs.cells["x"].to_numpy(), inputs.cells["y"].to_numpy())
    rows = []
    for kind in KINDS:
        facilities = inputs.of_kind(kind)
        normal = reach(roads, facilities)
        found = reach(roads, facilities, avoid) if len(avoid) else normal
        unreachable = np.array([not np.isfinite(found.to(n)) for n in cell_nodes], dtype=bool)
        cut_off = inputs.cells.loc[unreachable].groupby("ward_id")["people"].sum()
        for ward_id, origin in start.iterrows():
            node, snap = origin["node"], origin["snap"]
            ok = np.isfinite(found.to(node))
            usual = normal.to(node)
            # no safe route: keep the usual route, so the map can show where it crosses danger
            way = found if ok else normal
            facility_id = way.facility.get(node)
            rows.append(
                {
                    "run_id": run_id,
                    "ward_id": ward_id,
                    "kind": kind,
                    "facility_id": facility_id,
                    "facility_name": _name(names, facility_id),
                    "distance_m": round(found.to(node) + snap) if ok else None,
                    "baseline_m": round(usual + snap) if np.isfinite(usual) else None,
                    "status": (RouteStatus.OK if ok else RouteStatus.NO_SAFE_ROUTE).value,
                    "avoided_km": float(avoided_km.get(ward_id, 0.0)),
                    "people_cut_off": int(round(float(cut_off.get(ward_id, 0.0)))),
                    "line": route_line(roads, way.paths[node]) if node in way.paths else None,
                }
            )
    return rows


def _name(names: pd.Series, facility_id) -> str | None:
    if facility_id is None:
        return None
    name = names.get(facility_id)
    return name if isinstance(name, str) and name else None


def save_routes(rows: list[dict]) -> None:
    init_db()
    run_ids = {r["run_id"] for r in rows}
    now = datetime.now(UTC)
    with SessionLocal.begin() as session:
        session.execute(delete(WardRoute).where(WardRoute.run_id.in_(run_ids)))
        for r in rows:
            fields = {k: v for k, v in r.items() if k != "line"}
            line = r["line"]
            session.add(
                WardRoute(
                    **fields,
                    computed_at=now,
                    geometry=f"SRID=4326;{line.wkt}" if line is not None else None,
                )
            )


def delete_routes(run_id: str) -> None:
    with SessionLocal.begin() as session:
        session.execute(delete(WardRoute).where(WardRoute.run_id == run_id))


def routes_summary(rows: list[dict]) -> str:
    """One line for the run, then one per facility kind (each starting with '- ')."""
    table = pd.DataFrame(rows)
    avoided = table.groupby("ward_id")["avoided_km"].first().sum()
    lines = [f"Run {table['run_id'].iloc[0]}: {avoided:.1f} km of road avoided."]
    for kind in KINDS:
        part = table[table["kind"] == kind]
        blocked = part[part["status"] == RouteStatus.NO_SAFE_ROUTE.value]["ward_id"].tolist()
        longer = int(((part["distance_m"] - part["baseline_m"]).fillna(0) > 0).sum())
        lines.append(
            f"- {kind}: median {part['distance_m'].median() / 1000:.2f} km by road; longer than "
            f"usual from {longer} wards; wards with no safe route: {', '.join(blocked) or 'none'}; "
            f"{int(part['people_cut_off'].sum()):,} people cannot reach one."
        )
    return "\n".join(lines)


# --- Report ---------------------------------------------------------------------------------


def _row(cells_: list) -> str:
    return "| " + " | ".join(str(c) for c in cells_) + " |"


def report(inputs: Inputs, table: pd.DataFrame, demo_rows: list[dict], command: str) -> str:
    f = inputs.facilities
    dropped = f.attrs.get("dropped", {})
    shelters = inputs.of_kind("shelter")
    snapped = {
        kind: int(
            (
                inputs.roads.nearest(
                    *TO_METRIC.transform(inputs.of_kind(kind)["lon"], inputs.of_kind(kind)["lat"])
                )[1]
                <= MAX_SNAP_M
            ).sum()
        )
        for kind in KINDS
    }
    total = int(table["population"].sum())
    edges = inputs.roads.edges
    lines = [
        "# Vulnerability, impact and routes (Stage 12)",
        "",
        f"Built {date.today():%d %b %Y} with `{command}`. Method in `models/impact.py`.",
        "",
        "## Inputs",
        "",
        f"- **Population**: Meta/CIESIN HRSL, about 30 m cells. {len(inputs.cells):,} populated "
        f"cells inside the 24 wards, **{total:,} people** in all. For comparison, Census 2011 "
        "counted 12,442,373 in Greater Mumbai: HRSL spreads census counts over buildings and "
        "is an estimate of where people live, not a new count.",
        f"- **Hospitals**: {int((f['kind'] == 'hospital').sum()):,} OSM amenity=hospital "
        f"({int(f.loc[f['kind'] == 'hospital', 'emergency'].sum())} tagged emergency=yes), "
        f"{snapped['hospital']:,} on the road network. Many are small nursing homes. Left out: "
        f"{dropped.get('specialist', 0)} named as specialist clinics or labs (eye, dental, "
        f"diagnostic and similar) and {dropped.get('unnamed', 0)} facilities of all kinds with "
        "no name, since a route cannot point anyone to them.",
        f"- **Shelter candidates**: {len(shelters):,} ("
        + ", ".join(
            f"{n} {SHELTER_KINDS[k]}{'s' if n != 1 else ''}"
            for k, n in shelters["kind"].value_counts().items()
        )
        + f"), {snapped['shelter']:,} on the road network. "
        + (
            "From an official list in data/raw/shelters/."
            if shelters["official"].any()
            else "Not an official list: BMC opens municipal schools as temporary shelters, but "
            "the list is not open data."
        ),
        f"- **Roads**: OSM drive network, {len(inputs.roads.nodes):,} junctions and "
        f"{len(edges):,} segments ({edges['length'].sum() / 1000:,.0f} km). "
        f"{edges['low_lying'].mean():.0%} of segments have their midpoint below "
        f"{LOW_LYING_M:.0f} m; {int(edges['near_spot'].sum())} are within {SPOT_RADIUS_M} m of "
        f"one of {len(waterlogging_spots())} reported waterlogging spots.",
        "",
        "## Vulnerability per ward",
        "",
        "Index = mean of four ranks among the 24 wards (0 lowest, 1 highest): "
        + "; ".join(INDICATORS.values())
        + ". It orders wards and does not measure absolute need.",
        "",
        _row(
            [
                "Ward",
                "People",
                "Per km²",
                "Low-lying share",
                "To shelter (km)",
                f"> {SHELTER_FAR_KM:g} km from shelter",
                "To hospital (km)",
                f"> {HOSPITAL_FAR_KM:g} km from hospital",
                "Index",
            ]
        ),
        _row(["---"] + ["---:"] * 8),
    ]
    for ward_id, r in table.sort_values("vulnerability_index", ascending=False).iterrows():
        lines.append(
            _row(
                [
                    ward_id,
                    f"{int(r['population']):,}",
                    f"{r['density_per_km2']:,.0f}",
                    f"{r['low_lying_share']:.0%}",
                    f"{r['shelter_km']:.2f}",
                    f"{r['shelter_far_share']:.0%}",
                    f"{r['hospital_km']:.2f}",
                    f"{r['hospital_far_share']:.0%}",
                    f"{r['vulnerability_index']:.2f}",
                ]
            )
        )
    lines += [
        "",
        "Distances are along roads, weighted by where people live. Low-lying share uses the "
        "Copernicus surface model, whose heights include buildings, so it undercounts low "
        "ground in dense areas.",
        "",
    ]
    if demo_rows:
        lines += _route_section(demo_rows, table)
    lines += [
        "## Limits",
        "",
        "- Shelter candidates are places that could serve as shelters, not designated ones; "
        "capacity is unknown. Hospital tags do not say which take emergencies.",
        "- Roads are avoided by ward level and terrain, not by observed water on the road.",
        "- The vulnerability index has no social indicators (income, age, housing, slums): no "
        "open per-ward data were found.",
        "- Routes start at one point per ward. Footpaths are not in the network.",
        "",
    ]
    return "\n".join(lines)


def _route_section(rows: list[dict], table: pd.DataFrame) -> list[str]:
    run_id = rows[0]["run_id"]
    _, levels = run_levels(run_id)
    out = [
        f"## Routes in a flood run: `{run_id}`",
        "",
        "Segments avoided: in orange or red wards, midpoint below "
        f"{LOW_LYING_M:.0f} m or within {SPOT_RADIUS_M} m of a waterlogging spot. "
        '"No safe route" means every way out crosses an avoided segment; the map then shows '
        "the usual route in red. People cut off: people whose nearest junction cannot reach any "
        "shelter candidate without an avoided segment.",
        "",
    ]
    hot = sum(level in AVOID_LEVELS for level in levels.values())
    if hot == len(levels):
        out += [
            f"Every ward is {'/'.join(sorted({lv.value for lv in levels.values()}))} in this "
            "run (the Stage 5 flood model over-warns, docs/flood_model.md), so this is the worst "
            "case: low-lying roads are avoided across the whole city.",
            "",
        ]
    out += [
        _row(
            [
                "Ward",
                "Level",
                "Road avoided (km)",
                "Nearest shelter candidate",
                "km (usual)",
                "Nearest hospital",
                "km (usual)",
                "People cut off from a shelter",
            ]
        ),
        _row(["---", "---", "---:", "---", "---:", "---", "---:", "---:"]),
    ]

    def km(r) -> str:
        if r["status"] != RouteStatus.OK.value:
            usual = f" (usual {r['baseline_m'] / 1000:.2f})" if r["baseline_m"] else ""
            return f"no safe route{usual}"
        usual = f" ({r['baseline_m'] / 1000:.2f})" if r["baseline_m"] != r["distance_m"] else ""
        return f"{r['distance_m'] / 1000:.2f}{usual}"

    by = {(r["ward_id"], r["kind"]): r for r in rows}
    rank = {level: i for i, level in enumerate(Severity)}
    order = sorted(table.index, key=lambda w: (-rank[levels.get(w, Severity.GREEN)], w))
    for ward_id in order:
        s, h = by[(ward_id, "shelter")], by[(ward_id, "hospital")]
        out.append(
            _row(
                [
                    ward_id,
                    levels.get(ward_id, Severity.GREEN).value,
                    f"{s['avoided_km']:.1f}",
                    s["facility_name"] or "(unnamed)",
                    km(s),
                    h["facility_name"] or "(unnamed)",
                    km(h),
                    f"{s['people_cut_off']:,}",
                ]
            )
        )
    out += ["", routes_summary(rows), ""]
    return out


def latest_flood_run() -> str | None:
    """The newest flood run with an orange or red ward (the report's route example)."""
    with SessionLocal() as session:
        row = session.execute(
            text(
                "SELECT w.run_id FROM ward_risk w JOIN risk_runs r ON r.run_id = w.run_id "
                "WHERE w.hazard = 'flood' AND w.severity IN ('orange', 'red') "
                "GROUP BY w.run_id, r.issued_at ORDER BY r.issued_at DESC LIMIT 1"
            )
        ).first()
    return row[0] if row else None


# --- CLI ------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 12: vulnerability, impact and routes.")
    parser.add_argument("--fetch", action="store_true", help="population, facilities, roads")
    parser.add_argument("--build", action="store_true", help="vulnerability per ward -> PostGIS")
    parser.add_argument("--report", metavar="PATH", help="write the Markdown report (with --build)")
    parser.add_argument(
        "--example-run", metavar="RUN_ID", help="flood run for the report's route table"
    )
    parser.add_argument(
        "--routes", metavar="RUN_ID", nargs="+", help="routes for runs, or baseline"
    )
    args = parser.parse_args(argv)
    if not (args.fetch or args.build or args.routes):
        parser.error("give --fetch, --build or --routes")

    try:
        if args.fetch:
            fetch()
        if not (args.build or args.routes):
            return 0
        inputs = Inputs.load()
        if args.build:
            table = build(inputs)
            save_vulnerability(table, inputs.facilities)
            baseline = routes(inputs, BASELINE)
            save_routes(baseline)
            print(
                f"Saved vulnerability for {len(table)} wards, {len(inputs.facilities)} "
                f"facilities and baseline routes."
            )
            if args.report:
                example = args.example_run or latest_flood_run()
                demo_rows = []
                if example:
                    demo_rows = routes(inputs, example)
                    save_routes(demo_rows)
                command = "python -m models.impact --build --report " + Path(args.report).as_posix()
                if args.example_run:
                    command += f" --example-run {args.example_run}"
                out = Path(args.report)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(report(inputs, table, demo_rows, command), encoding="utf-8")
                print(f"Wrote {out}")
        for run_id in args.routes or []:
            rows = routes(inputs, run_id)
            save_routes(rows)
            print(routes_summary(rows))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
