"""Offline tests for static ward features and the flood label table."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from pipelines import labels, static


def test_slope_of_a_tilted_plane():
    # Rises 1 m per 10 m going east: slope = atan(0.1) = 5.71 degrees everywhere.
    heights = np.tile(np.arange(20, dtype=float), (10, 1))
    slope = static.slope_degrees(heights, dx=10.0, dy=10.0)
    np.testing.assert_allclose(slope, np.degrees(np.arctan(0.1)), atol=1e-6)
    assert static.slope_degrees(np.full((5, 5), 3.0), 10, 10).max() == 0


def test_cell_size_shrinks_east_west_away_from_the_equator():
    transform = from_origin(72.0, 20.0, 0.001, 0.001)
    dx, dy = static.cell_size_m(transform, lat=19.0)
    assert dy == pytest.approx(111.32)
    assert dx == pytest.approx(111.32 * np.cos(np.radians(19.0)))


def test_distance_to_lines_and_ward_means():
    transform = from_origin(0, 10, 1, 1)  # 10 x 10 cells of 1 degree
    line = gpd.GeoDataFrame(geometry=[LineString([(0.5, 0), (0.5, 10)])], crs="EPSG:4326")
    distance = static.distance_to_lines(line, (10, 10), transform, dx=100.0, dy=100.0)
    np.testing.assert_allclose(distance[0], np.arange(10) * 100.0)

    wards = gpd.GeoDataFrame(geometry=[box(0, 0, 5, 10), box(5, 0, 10, 10)], crs="EPSG:4326")
    zones = static.ward_labels(wards, (10, 10), transform)
    assert (zones[:, :5] == 1).all() and (zones[:, 5:] == 2).all()

    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    assert np.isnan(static.distance_to_lines(empty, (10, 10), transform, 100, 100)).all()


def row(**changes) -> dict:
    base = {
        "date": "2024-07-08",
        "ward_id": "F-S",
        "flooded": "1",
        "location": "Hindmata junction",
        "lat": "",
        "lon": "",
        "evidence": "news",
        "source_url": "https://example.org/report",
        "confidence": "medium",
        "reviewed": "no",
        "notes": "",
    }
    return base | changes


def test_a_good_row_has_no_problems():
    assert labels.problems(pd.DataFrame([row()])) == []


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"date": "8 July 2024"}, "not YYYY-MM-DD"),
        ({"ward_id": "F/S"}, "unknown ward_id"),
        ({"flooded": "yes"}, "0 or 1"),
        ({"evidence": "rumour"}, "evidence"),
        ({"source_url": ""}, "needs a source_url"),
        ({"evidence": "dry_day"}, "cannot be flooded"),
        ({"lat": "19.0"}, "both lat and lon"),
        ({"reviewed": ""}, "yes or no"),
    ],
)
def test_bad_rows_are_reported(changes, message):
    found = labels.problems(pd.DataFrame([row(**changes)]))
    assert any(message in m for m in found), found


def test_point_must_be_inside_the_named_ward():
    wards = gpd.GeoDataFrame(
        {"ward_id": ["F-S", "F-N"]},
        geometry=[box(72.8, 19.0, 72.9, 19.1), box(72.8, 19.1, 72.9, 19.2)],
    )
    good = pd.DataFrame([row(lat="19.05", lon="72.85")])
    wrong = pd.DataFrame([row(lat="19.15", lon="72.85")])
    assert labels.problems(good, wards) == []
    assert "not F-S" in labels.problems(wrong, wards)[0]


def test_a_ward_day_cannot_be_both_flooded_and_dry():
    table = pd.DataFrame([row(), row(flooded="0", evidence="dry_day", source_url="")])
    assert any("both flooded and not flooded" in m for m in labels.problems(table))


def test_the_label_table_in_the_repo_is_valid_and_large_enough():
    table = labels.read()
    assert labels.problems(table, labels.load_wards()) == []
    days = labels.ward_days(table)
    assert len(days) >= labels.MIN_WARD_DAYS
    assert (days["flooded"] == 1).any() and (days["flooded"] == 0).any()
