"""Offline tests for the raw store, the pilot grid and harmonising. No network needed."""

import json
from datetime import date

import numpy as np
import pytest

from api.config import Settings
from pipelines import grid, store
from pipelines.sources import openmeteo

DAY = date(2025, 7, 20)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    fake = Settings("unused", "unused", (72.70, 18.85, 73.10, 19.35), tmp_path)
    for module in (store, grid):
        monkeypatch.setattr(module, "settings", fake)
    return tmp_path


def test_target_grid_covers_pilot_box():
    lons, lats = grid.target_coords()
    assert lons.tolist() == [72.7, 72.8, 72.9, 73.0, 73.1]
    assert lats.tolist() == [18.85, 18.95, 19.05, 19.15, 19.25, 19.35]


@pytest.mark.parametrize("lat_ascending", [True, False])
def test_regrid_reproduces_a_linear_field(lat_ascending):
    src_lons = np.arange(72.0, 74.01, 0.25)
    src_lats = np.arange(18.0, 20.01, 0.25)
    if not lat_ascending:
        src_lats = src_lats[::-1]
    field = src_lats[:, None] * 10 + src_lons[None, :]
    out = grid.regrid(field, src_lons, src_lats, "bilinear")
    lons, lats = grid.target_coords()
    np.testing.assert_allclose(out, lats[:, None] * 10 + lons[None, :], atol=1e-3)


def test_regrid_average_keeps_mean_and_skips_missing():
    src_lons = np.arange(72.025, 74.0, 0.05)
    src_lats = np.arange(18.025, 20.0, 0.05)
    field = np.full((src_lats.size, src_lons.size), 8.0)
    field[::2, ::2] = np.nan
    out = grid.regrid(field, src_lons, src_lats, "average")
    np.testing.assert_allclose(out, 8.0, atol=1e-4)


def test_raw_files_are_logged_and_never_overwritten(tmp_data):
    path = store.write_raw("demo", DAY, "a.json", b"{}", "http://example.test/a")
    assert path == tmp_data / "raw" / "demo" / "2025" / "07" / "20" / "a.json"
    entry = json.loads((tmp_data / "raw" / store.MANIFEST).read_text().splitlines()[0])
    assert entry["sha256"] == store.sha256(path) and entry["bytes"] == 2
    with pytest.raises(FileExistsError):
        store.write_raw("demo", DAY, "a.json", b"changed", "http://example.test/a")
    assert path.read_bytes() == b"{}"


def test_openmeteo_harmonize_puts_points_in_the_right_cells(tmp_data):
    lats, lons = openmeteo._points()
    times = [f"2025-07-20T{h:02d}:00" for h in range(24)]
    points = []
    for lat, lon in zip(lats, lons, strict=True):
        hourly = {"time": times}
        for var in openmeteo.VARIABLES:
            for model in openmeteo.MODELS:
                hourly[f"{var}_{model}"] = [lat * 10 + lon] * 24
        points.append({"latitude": lat, "longitude": lon, "hourly": hourly})
    del points[0]["hourly"]["precipitation_gfs_seamless"]  # one model missing at one point
    store.write_raw("openmeteo", DAY, "forecast_x.json", json.dumps(points).encode(), "test")

    ds = openmeteo.harmonize(DAY)
    n = len(openmeteo.MODELS)
    assert dict(ds.sizes) == {"time": 24, "lat": 6, "lon": 5, "model": n}
    cell = ds["temperature_2m"].sel(model="ecmwf_ifs025", lat=19.15, lon=72.9).isel(time=0)
    assert float(cell) == pytest.approx(19.15 * 10 + 72.9, abs=1e-3)
    assert int(ds["precipitation"].isnull().sum()) == 24
    assert ds["wind_speed_10m"].attrs["units"] == "m s-1"


def test_openmeteo_harmonize_keeps_only_the_models_an_older_file_has(tmp_data):
    lats, lons = openmeteo._points()
    times = [f"2025-07-20T{h:02d}:00" for h in range(24)]
    older = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]  # files fetched before Stage 11
    points = [
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": {"time": times}
            | {f"{v}_{m}": [1.0] * 24 for v in openmeteo.VARIABLES for m in older},
        }
        for lat, lon in zip(lats, lons, strict=True)
    ]
    store.write_raw("openmeteo", DAY, "forecast_x.json", json.dumps(points).encode(), "test")
    ds = openmeteo.harmonize(DAY)
    assert list(ds["model"].values) == older
    assert int(ds["precipitation"].isnull().sum()) == 0
