"""Stage 9: heat stress. The last group needs the database (docker compose up)."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from api import main
from api.schemas import Hazard, Severity, Trust, WardRisk
from models import heat, risk

LIMITS = {"yellow": 36.9, "orange": 37.8, "red": 39.4}


def test_apparent_temperature_rises_with_humidity_and_falls_with_wind():
    # 30 degC, 70% humidity, 2 m/s: vapour pressure 29.6 hPa -> 30 + 9.77 - 1.4 - 4 = 34.4
    assert heat.apparent_temperature(30.0, 70.0, 2.0) == pytest.approx(34.37, abs=0.02)
    assert heat.apparent_temperature(30.0, 90.0, 2.0) > heat.apparent_temperature(30.0, 50.0, 2.0)
    assert heat.apparent_temperature(30.0, 70.0, 6.0) < heat.apparent_temperature(30.0, 70.0, 1.0)


def test_levels_start_at_each_threshold():
    assert heat.level(36.89, LIMITS) == Severity.GREEN
    assert heat.level(36.9, LIMITS) == Severity.YELLOW
    assert heat.level(37.8, LIMITS) == Severity.ORANGE
    assert heat.level(39.4, LIMITS) == Severity.RED


def test_daily_max_uses_the_ist_day_and_needs_all_24_hours():
    hours = pd.date_range("2025-01-01T19:00", "2025-01-03T05:00", freq="h")  # UTC
    values = pd.DataFrame({"c": 30.0}, index=hours)
    values.loc[pd.Timestamp("2025-01-02T18:00"), "c"] = 40.0  # 23:30 IST on 2 Jan
    values.loc[pd.Timestamp("2025-01-02T19:00"), "c"] = 45.0  # 00:30 IST on 3 Jan
    daily = heat.daily_max(values)
    assert daily.at[date(2025, 1, 2), "c"] == 40.0
    assert np.isnan(daily.at[date(2025, 1, 3), "c"])  # only 11 hours of 3 Jan


def test_quantile_map_shifts_and_keeps_the_offset_outside_the_training_range():
    mapping = heat.fit_map(np.arange(30.0, 40.0, 0.1), np.arange(31.0, 41.0, 0.1))
    out = heat.apply_map(mapping, np.array([35.0, 45.0, 20.0, np.nan]))
    assert out[:3] == pytest.approx([36.0, 46.0, 21.0], abs=0.05)
    assert np.isnan(out[3])


def test_median_candidate_is_the_middle_model():
    identity = {"forecast": [0.0, 100.0], "era5": [0.0, 100.0]}
    maps = {m: identity for m in [*heat.MODELS, "mean"]}
    values = pd.DataFrame([[36.0, 38.0, 41.0]], columns=heat.MODELS)
    out = heat.combine(values, maps).iloc[0]
    assert out["corrected_median"] == pytest.approx(38.0)
    assert out["corrected_mean"] == pytest.approx(115 / 3)
    assert out["corrected_ecmwf"] == pytest.approx(36.0)


def test_scores_count_each_cell_day_once():
    rows = pd.DataFrame(
        {
            "date": [date(2025, 4, 1)] * 3,
            "cell": ["19.00N", "19.00N", "19.25N"],
            "f": [37.0, 39.0, 36.0],  # the first cell averages 38.0: a hit for orange
            "era5": [38.5, 38.5, 36.0],
        }
    )
    out = heat.scores(rows, LIMITS, {"f": "f"}).iloc[0]
    assert (out["orange_hits"], out["orange_misses"], out["orange_false"]) == (1, 0, 0)
    assert out["mae"] == pytest.approx(0.25)


def test_choose_prefers_orange_csi_among_candidates_only():
    table = pd.DataFrame(
        [
            {"key": "corrected_mean", "orange_csi": 0.3, "yellow_csi": 0.5, "mae": 1.0},
            {"key": "corrected_ecmwf", "orange_csi": 0.4, "yellow_csi": 0.4, "mae": 1.2},
            {"key": "persistence", "orange_csi": 0.9, "yellow_csi": 0.9, "mae": 0.5},
        ]
    )
    assert heat.choose(table) == "corrected_ecmwf"


def test_forecast_run_levels_probability_and_test_flag(monkeypatch):
    identity = {"forecast": [0.0, 100.0], "era5": [0.0, 100.0]}
    model = {
        "created_at": "x",
        "thresholds": LIMITS,
        "method": "corrected_median",
        "maps": {m: identity for m in [*heat.MODELS, "mean"]},
    }
    fc = pd.DataFrame(
        [[36.0, 38.0, 40.0], [30.0, 31.0, 32.0]], index=["A", "B"], columns=heat.MODELS
    )
    issued = datetime(2026, 4, 20, 5, tzinfo=UTC)
    monkeypatch.setattr(heat, "load_model", lambda: model)
    monkeypatch.setattr(heat, "forecast_daily", lambda day: fc)
    monkeypatch.setattr(risk, "forecast_issued", lambda day: issued)
    now = issued + timedelta(hours=1)
    result = heat.compute(date(2026, 4, 20), now=now)
    by_ward = {r.ward_id: r for r in result["risks"]}
    assert by_ward["A"].severity == Severity.ORANGE and by_ward["A"].value == 38.0
    assert by_ward["A"].probability == pytest.approx(2 / 3, abs=1e-3)  # 2 of 3 models at orange
    assert by_ward["B"].severity == Severity.GREEN and by_ward["B"].probability == 0
    assert by_ward["A"].unit == "°C" and by_ward["A"].data_age_minutes == 60
    assert result["run"]["is_test"] and result["run"]["run_id"].startswith("heat-forecast-")
    assert heat.compute(date(2026, 4, 20), now=now, live=True)["run"]["is_test"] is False


def test_heat_warnings_name_the_hazard():
    run = main.RunInfo(
        run_id="h",
        hazard="heat",
        issued_at=datetime(2026, 4, 20, tzinfo=UTC),
        data_time=datetime(2026, 4, 20, tzinfo=UTC),
        data_age_minutes=30,
        rain_source="forecast",
        model="heat_at",
        is_test=False,
        note=None,
        valid_from=datetime(2026, 4, 19, 18, 30, tzinfo=UTC),
        valid_to=datetime(2026, 4, 20, 18, 30, tzinfo=UTC),
    )
    risks = [
        WardRisk(
            ward_id="A",
            hazard=Hazard.HEAT,
            valid_from=run.valid_from,
            valid_to=run.valid_to,
            probability=1.0,
            severity=Severity.RED,
            trust=Trust.MEDIUM,
            run_id="h",
            value=40.1,
            unit="°C",
        )
    ]
    assert main.warnings_for(run, risks, datetime(2026, 4, 20, 6, tzinfo=UTC)) == [
        "Heat risk RED: A."
    ]
    assert main.warnings_for(None, [], datetime(2026, 4, 20, tzinfo=UTC), Hazard.HEAT)[0].endswith(
        "python -m models.heat --date today"
    )


# --- With the database ----------------------------------------------------------------------

RUN_ID = "pytest-stage9-heat"


@pytest.fixture(scope="module")
def client():
    """A heat run dated 2000, removed afterwards."""
    issued = datetime(2000, 4, 2, tzinfo=UTC)
    valid_from, valid_to = risk.valid_window(date(2000, 4, 1))
    result = {
        "run": {
            "run_id": RUN_ID,
            "issued_at": issued,
            "data_time": issued,
            "rain_source": "era5",
            "model": "pytest",
            "is_test": True,
            "note": "pytest",
        },
        "risks": [
            WardRisk(
                ward_id=w,
                hazard=Hazard.HEAT,
                valid_from=valid_from,
                valid_to=valid_to,
                probability=p,
                severity=heat.level(v, LIMITS),
                trust=Trust.MEDIUM,
                run_id=RUN_ID,
                value=v,
                unit="°C",
            )
            for w, p, v in [("G-N", 1.0, 39.6), ("A", 0.0, 36.0)]
        ],
    }
    try:
        risk.delete_run(RUN_ID)
        risk.save(result)
    except OperationalError:
        pytest.skip("database not running")
    yield TestClient(main.app)
    risk.delete_run(RUN_ID)


def test_heat_run_through_the_api(client):
    body = client.get("/v1/wards", params={"run_id": RUN_ID}).json()
    props = {f["properties"]["ward_id"]: f["properties"] for f in body["features"]}
    assert body["hazard"] == "heat" and body["run"]["hazard"] == "heat"
    assert props["G-N"]["severity"] == "red" and props["G-N"]["value"] == 39.6
    assert props["G-N"]["unit"] == "°C"
    assert "Heat risk RED: G-N." in body["warnings"]

    point = client.get(
        "/v1/risk", params={"lat": 19.0178, "lon": 72.8426, "hazard": "heat", "run_id": RUN_ID}
    ).json()
    assert point["risks"][0]["hazard"] == "heat" and point["risks"][0]["value"] == 39.6

    heat_runs = client.get("/v1/runs", params={"hazard": "heat", "limit": 200}).json()
    flood_runs = client.get("/v1/runs", params={"hazard": "flood", "limit": 200}).json()
    assert RUN_ID in {r["run_id"] for r in heat_runs}
    assert RUN_ID not in {r["run_id"] for r in flood_runs}


def test_flood_stays_the_default(client):
    body = client.get("/v1/wards").json()
    assert body["hazard"] == "flood"
    assert body["run"] is None or body["run"]["hazard"] in (None, "flood")
