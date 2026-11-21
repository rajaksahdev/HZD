"""Stage 12: vulnerability, impact and routes. Offline: a toy road graph and synthetic cells;
the API part needs the database and skips without it."""

from datetime import UTC, date, datetime, timedelta

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from shapely.geometry import LineString, box
from sqlalchemy.exc import OperationalError

from api import main
from api.schemas import Hazard, Route, RouteStatus, Severity, Trust, WardRisk
from models import impact, risk

# --- A toy road network ---------------------------------------------------------------------
#
#   A ---- B ---- C (shelter)        ward "X" holds A, B, C; ward "Y" holds D
#    \           /
#     `--- D ---'                    A-B-C is 2 km, A-D-C is 3 km

LON0, LAT0 = 72.85, 19.05
DEG = 1 / 111.0  # about 1 km of latitude


def toy_roads(low_lying=("B", "C")) -> impact.Roads:
    coords = {
        "A": (LON0, LAT0),
        "B": (LON0, LAT0 + DEG),
        "C": (LON0, LAT0 + 2 * DEG),
        "D": (LON0 + DEG, LAT0 + DEG),
    }
    graph = nx.Graph()
    for n, (x, y) in coords.items():
        graph.add_node(n, x=x, y=y)
    lengths = {("A", "B"): 1000, ("B", "C"): 1000, ("A", "D"): 1500, ("D", "C"): 1500}
    for (u, v), length in lengths.items():
        graph.add_edge(u, v, length=float(length), geometry=None)
    nodes = np.array(list(coords))
    lonlat = np.array([coords[n] for n in nodes])
    xy = np.column_stack(impact.TO_METRIC.transform(lonlat[:, 0], lonlat[:, 1]))
    edges = pd.DataFrame(
        [(u, v, length) for (u, v), length in lengths.items()], columns=["u", "v", "length"]
    )
    edges["ward_id"] = ["X", "X", "Y", "Y"]
    edges["low_lying"] = [(u in low_lying and v in low_lying) for u, v in lengths]
    edges["near_spot"] = False
    return impact.Roads(graph, nodes, xy, lonlat, edges, cKDTree(xy))


def shelter_at(node: str, roads: impact.Roads) -> pd.DataFrame:
    lon, lat = roads.lonlat[list(roads.nodes).index(node)]
    return pd.DataFrame({"facility_id": [f"f-{node}"], "lon": [lon], "lat": [lat]})


def test_reach_gives_the_road_distance_to_the_nearest_facility():
    roads = toy_roads()
    found = impact.reach(roads, shelter_at("C", roads))
    assert found.to("A") == pytest.approx(2000, abs=1)
    assert found.paths["A"] == ["A", "B", "C"]
    assert found.facility["A"] == "f-C"


def test_avoided_roads_force_a_detour_and_then_cut_off():
    roads = toy_roads()
    shelters = shelter_at("C", roads)
    avoid = roads.avoided({"X": Severity.RED, "Y": Severity.GREEN}, Hazard.FLOOD)
    assert set(zip(avoid["u"], avoid["v"], strict=True)) == {("B", "C")}
    found = impact.reach(roads, shelters, avoid)
    assert found.paths["A"] == ["A", "D", "C"]
    assert found.to("A") == pytest.approx(3000, abs=1)

    roads = toy_roads(low_lying=("A", "B", "C", "D"))
    avoid = roads.avoided({"X": Severity.RED, "Y": Severity.ORANGE}, Hazard.FLOOD)
    assert len(avoid) == 4
    found = impact.reach(roads, shelters, avoid)
    assert not np.isfinite(found.to("A"))  # every way out crosses an avoided road
    assert found.to("C") == pytest.approx(0, abs=1)  # the shelter's own junction


def test_nothing_is_avoided_in_yellow_wards_or_heat_runs():
    roads = toy_roads(low_lying=("A", "B", "C", "D"))
    assert roads.avoided({"X": Severity.YELLOW, "Y": Severity.YELLOW}, Hazard.FLOOD).empty
    assert roads.avoided({"X": Severity.RED, "Y": Severity.RED}, Hazard.HEAT).empty


def test_reach_does_not_change_the_shared_graph():
    roads = toy_roads()
    before = roads.graph.number_of_nodes()
    impact.reach(roads, shelter_at("C", roads), roads.edges.iloc[:1])
    assert roads.graph.number_of_nodes() == before
    assert impact.SOURCE not in roads.graph


def test_facilities_too_far_from_any_road_are_ignored():
    roads = toy_roads()
    far = pd.DataFrame({"facility_id": ["sea"], "lon": [LON0 - 0.05], "lat": [LAT0]})
    assert impact.reach(roads, far).distance == {}


def test_route_line_runs_from_the_start():
    roads = toy_roads()
    line = impact.route_line(roads, ["C", "B", "A"])
    assert line.coords[0] == pytest.approx(tuple(roads.lonlat[2]))
    assert line.coords[-1] == pytest.approx(tuple(roads.lonlat[0]))
    roads.graph["A"]["B"]["geometry"] = LineString(
        [roads.lonlat[0], (LON0 - 0.001, LAT0 + DEG / 2), roads.lonlat[1]]
    )
    line = impact.route_line(roads, ["B", "A"])  # stored A->B, walked B->A
    assert line.coords[0] == pytest.approx(tuple(roads.lonlat[1]))
    assert len(line.coords) == 3
    assert impact.route_line(roads, ["A"]) is None


# --- Facilities -----------------------------------------------------------------------------


def test_parse_facilities_keeps_known_kinds_and_way_centres():
    elements = [
        {"type": "node", "id": 1, "lat": 19.0, "lon": 72.8, "tags": {"amenity": "hospital"}},
        {
            "type": "way",
            "id": 2,
            "center": {"lat": 19.1, "lon": 72.9},
            "tags": {"amenity": "school", "name": "BMC School"},
        },
        {"type": "node", "id": 3, "lat": 19, "lon": 72.8, "tags": {"emergency": "assembly_point"}},
        {"type": "node", "id": 4, "lat": 19.0, "lon": 72.8, "tags": {"amenity": "bench"}},
        {"type": "relation", "id": 5, "tags": {"amenity": "school"}},  # no position
    ]
    table = impact.parse_facilities(elements)
    assert table["facility_id"].tolist() == ["osm-node-1", "osm-way-2", "osm-node-3"]
    assert table["kind"].tolist() == ["hospital", "school", "assembly_point"]
    assert table.loc[1, "lat"] == 19.1


@pytest.mark.parametrize(
    "name, specialist",
    [
        ("Krishna Eye Care Centre", True),
        ("SRL Diagnostics Andheri", True),
        ("gr8 smile dental clinic", True),
        ("Bombay Hospital and Medical Research Center", False),
        ("Sangita Maternity Surgical and Diagnostic Centre", False),
        ("Kalina Hospital", False),
        ("Competent Hospital", False),  # "pet" only as a whole word
    ],
)
def test_specialist_clinics_are_told_from_hospitals(name, specialist):
    assert impact.specialist(name) is specialist


def test_unnamed_facilities_and_specialist_hospitals_are_not_usable():
    table = pd.DataFrame(
        {
            "name": ["Kalina Hospital", "Ojas Eye Clinic", None, "Eye Care School", " "],
            "kind": ["hospital", "hospital", "school", "school", "community_centre"],
        }
    )
    assert impact.usable(table).tolist() == [True, False, False, True, False]


# --- Vulnerability and impact ---------------------------------------------------------------


def test_ranks_run_from_zero_to_one():
    assert impact.ranked(pd.Series([5.0, 1.0, 3.0])).tolist() == [1.0, 0.0, 0.5]
    assert impact.ranked(pd.Series([2.0, 2.0])).tolist() == [0.5, 0.5]


def test_vulnerability_index_is_the_mean_rank():
    table = pd.DataFrame(
        {
            "density_per_km2": [3, 2, 1],
            "low_lying_share": [0.3, 0.2, 0.1],
            "shelter_km": [3, 2, 1],
            "hospital_km": [1, 2, 3],
        },
        index=["P", "Q", "R"],
    )
    index = impact.vulnerability_index(table)
    assert index.tolist() == [0.75, 0.5, 0.25]


def test_ward_table_sums_people_and_shares():
    wards = gpd.GeoDataFrame(
        {"ward_id": ["X"]}, geometry=[box(LON0, LAT0, LON0 + 0.01, LAT0 + 0.01)], crs=4326
    )
    cells = pd.DataFrame(
        {
            "people": [100.0, 300.0],
            "ward_id": ["X", "X"],
            "low_lying": [True, False],
        }
    )
    table = impact.ward_table(wards, cells, np.array([500.0, 1500.0]), np.array([np.inf, 1000.0]))
    row = table.loc["X"]
    assert row["population"] == 400
    assert row["low_lying_pop"] == 100 and row["low_lying_share"] == 0.25
    assert row["shelter_km"] == pytest.approx((100 * 0.5 + 300 * 1.5) / 400, abs=0.01)
    assert row["shelter_far_share"] == 0.75
    assert row["hospital_km"] == 1.0  # the unreachable cell is left out of the mean
    assert row["hospital_far_share"] == 0.25  # ... but counts as far
    assert row["area_km2"] == pytest.approx(1.1, abs=0.1)


# --- API helpers ----------------------------------------------------------------------------


def ward_risk(ward_id: str, p: float) -> WardRisk:
    start = datetime(2025, 8, 18, 18, 30, tzinfo=UTC)
    return WardRisk(
        ward_id=ward_id,
        hazard=Hazard.FLOOD,
        valid_from=start,
        valid_to=start + timedelta(days=1),
        probability=p,
        severity=risk.severity(p),
        trust=Trust.MEDIUM,
        run_id="r1",
    )


def run_info() -> main.RunInfo:
    t = datetime(2025, 8, 18, tzinfo=UTC)
    return main.RunInfo(
        run_id="r1",
        hazard="flood",
        issued_at=t,
        data_time=t,
        data_age_minutes=0,
        rain_source="forecast",
        model="m",
        is_test=True,
        note=None,
        valid_from=t,
        valid_to=t + timedelta(days=1),
    )


VULN = {
    "X": {"population": 1000, "low_lying_pop": 200, "vulnerability_index": 0.5},
    "Y": {"population": 500, "low_lying_pop": 0, "vulnerability_index": 1.0},
}


def test_impact_rows_use_low_lying_people_for_flood_and_everyone_for_heat():
    route = Route(ward_id="X", kind="shelter", run_id="r1", status=RouteStatus.OK, distance_m=800)
    rows = main.impact_rows(
        Hazard.FLOOD, run_info(), [ward_risk("X", 0.8)], VULN, {("X", "shelter"): route}
    )
    x, y = rows
    assert (x.exposed, x.people_at_risk, x.impact_score) == (200, 160, 160)
    assert x.shelter.distance_m == 800 and x.hospital is None
    assert y.probability is None and y.people_at_risk is None  # no risk row for Y
    heat = main.impact_rows(Hazard.HEAT, None, [], VULN, {})
    assert heat[0].exposed == 1000


def test_route_warnings_when_routes_ignore_the_run():
    run = run_info()
    red = [ward_risk("X", 0.9)]
    assert main.route_warnings(Hazard.FLOOD, run, red, "r1") == []
    assert "--routes r1" in main.route_warnings(Hazard.FLOOD, run, red, "baseline")[0]
    assert main.route_warnings(Hazard.FLOOD, run, [ward_risk("X", 0.1)], "baseline") == []
    assert main.route_warnings(Hazard.HEAT, run, red, "baseline") == []
    assert "--build" in main.route_warnings(Hazard.FLOOD, run, red, None)[0]


# --- With the database ----------------------------------------------------------------------

RUN_ID = "pytest-stage12"


@pytest.fixture(scope="module")
def client():
    """A red run for G-N dated 2000, with one saved route; removed afterwards."""
    from fastapi.testclient import TestClient

    issued = datetime(2000, 1, 2, tzinfo=UTC)
    valid_from, valid_to = risk.valid_window(date(2000, 1, 1))
    result = {
        "run": {
            "run_id": RUN_ID,
            "issued_at": issued,
            "data_time": issued,
            "rain_source": "observed",
            "model": "pytest",
            "is_test": True,
            "note": "pytest",
        },
        "risks": [
            WardRisk(
                ward_id="G-N",
                hazard=Hazard.FLOOD,
                valid_from=valid_from,
                valid_to=valid_to,
                probability=0.9,
                severity=Severity.RED,
                trust=Trust.LOW,
                run_id=RUN_ID,
            )
        ],
    }
    route = {
        "run_id": RUN_ID,
        "ward_id": "G-N",
        "kind": "shelter",
        "facility_id": "osm-node-1",
        "facility_name": "Test School",
        "distance_m": 900.0,
        "baseline_m": 600.0,
        "status": "ok",
        "avoided_km": 2.5,
        "people_cut_off": 10,
        "line": LineString([(72.84, 19.02), (72.845, 19.025)]),
    }
    try:
        risk.delete_run(RUN_ID)
        risk.save(result)
        impact.save_routes([route])
    except OperationalError:
        pytest.skip("database not running")
    yield TestClient(main.app)
    risk.delete_run(RUN_ID)


def test_impact_endpoint_uses_the_run_routes(client):
    body = client.get("/v1/impact", params={"run_id": RUN_ID}).json()
    assert body["routes_run_id"] == RUN_ID
    gn = next(w for w in body["wards"] if w["ward_id"] == "G-N")
    assert gn["probability"] == 0.9
    assert gn["shelter"]["facility_name"] == "Test School"
    if gn["population"] is not None:  # filled once models.impact --build has run
        assert gn["people_at_risk"] == round(0.9 * gn["exposed"])


def test_ward_impact_returns_route_geojson(client):
    body = client.get("/v1/wards/G-N/impact", params={"run_id": RUN_ID}).json()
    assert [w["ward_id"] for w in body["wards"]] == ["G-N"]
    lines = body["routes"]["features"]
    assert lines[0]["geometry"]["type"] == "LineString"
    assert lines[0]["properties"]["kind"] == "shelter"
    assert client.get("/v1/wards/ZZ/impact").status_code == 404


def test_deleting_a_run_deletes_its_routes(client):
    risk.delete_run(RUN_ID)
    body = client.get("/v1/impact", params={"hazard": "flood"}).json()
    assert body["routes_run_id"] != RUN_ID


def test_facilities_endpoint_is_geojson(client):
    body = client.get("/v1/facilities", params={"kind": "hospital"}).json()
    assert body["type"] == "FeatureCollection"
    assert all(f["properties"]["group"] == "hospital" for f in body["features"])
