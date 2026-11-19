"""Stage 6: risk run helpers and the API. The last group needs the database (docker compose up)."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from api import main
from api.schemas import Hazard, Severity, Trust, WardRisk
from models import flood, risk

NOW = datetime(2026, 7, 20, 6, 0, tzinfo=UTC)


def test_severity_bands():
    assert risk.severity(0.0) == Severity.GREEN
    assert risk.severity(0.25) == Severity.YELLOW
    assert risk.severity(0.5) == Severity.ORANGE
    assert risk.severity(0.749) == Severity.ORANGE
    assert risk.severity(0.75) == Severity.RED


def test_valid_window_is_the_ist_day():
    start, end = risk.valid_window(date(2025, 8, 19))
    assert start == datetime(2025, 8, 18, 18, 30, tzinfo=UTC)
    assert end - start == timedelta(days=1)


def test_forecast_mode_uses_imd_before_the_day_and_forecast_on_it(monkeypatch):
    day = date(2026, 7, 20)
    imd_days = {day - timedelta(days=3), day - timedelta(days=2), day}  # day - 1 not published
    monkeypatch.setattr(
        flood, "imd_at_wards", lambda d: pd.Series({"A": 1.0}) if d in imd_days else None
    )
    monkeypatch.setattr(risk, "forecast_at_wards", lambda d: pd.Series({"A": 2.0}))
    monkeypatch.setattr(risk, "forecast_issued", lambda d: datetime(2026, 7, 20, 5, tzinfo=UTC))
    rain = risk.RainInputs(day, "forecast")
    for offset in (-3, -2, -1, 0, 1):
        rain(day + timedelta(days=offset))
    assert [rain.used[day + timedelta(days=o)] for o in (-3, -2, -1, 0, 1)] == [
        "IMD",
        "IMD",
        "forecast",
        "forecast",
        "forecast",
    ]  # IMD exists for the day itself, but forecast mode must use the forecast there
    assert rain.data_time() == datetime(2026, 7, 20, 5, tzinfo=UTC)

    observed = risk.RainInputs(day, "observed")
    assert observed(day - timedelta(days=1)) is None  # a hindcast never falls back to forecast


def run_info(**changes) -> main.RunInfo:
    base = dict(
        run_id="r1",
        issued_at=NOW,
        data_time=NOW - timedelta(minutes=30),
        data_age_minutes=30,
        rain_source="forecast",
        model="flood_xgb",
        is_test=False,
        note=None,
        valid_from=NOW - timedelta(hours=6),
        valid_to=NOW + timedelta(hours=18),
    )
    return main.RunInfo(**{**base, **changes})


def ward(ward_id: str, level: Severity, trust: Trust = Trust.MEDIUM) -> WardRisk:
    return WardRisk(
        ward_id=ward_id,
        hazard=Hazard.FLOOD,
        valid_from=NOW - timedelta(hours=6),
        valid_to=NOW + timedelta(hours=18),
        probability=0.9,
        severity=level,
        trust=trust,
        run_id="r1",
    )


def test_no_warnings_for_a_fresh_quiet_live_run():
    assert main.warnings_for(run_info(), [ward("A", Severity.GREEN)], NOW) == []


def test_warnings_for_test_stale_old_and_red_runs():
    run = run_info(
        is_test=True,
        data_age_minutes=600,
        valid_from=NOW - timedelta(days=3),
        valid_to=NOW - timedelta(days=2),
    )
    risks = [ward("A", Severity.RED), ward("F-N", Severity.RED), ward("L", Severity.ORANGE)]
    out = main.warnings_for(run, risks, NOW)
    assert out[0].startswith("TEST data")
    assert any(w.startswith("Out of date") for w in out)
    assert "Stale data: the newest input is 10 hours old." in out
    assert "Flood risk RED: A, F-N." in out
    assert "Flood risk ORANGE: L." in out
    assert out.index("Flood risk RED: A, F-N.") < out.index("Flood risk ORANGE: L.")


def test_warnings_without_any_run_and_low_trust():
    assert main.warnings_for(None, [], NOW)[0].startswith("No risk run")
    out = main.warnings_for(run_info(), [ward("A", Severity.GREEN, Trust.LOW)], NOW)
    assert out and out[-1].startswith("Low trust")


def test_age_text():
    assert main.describe_age(45) == "45 minutes"
    assert main.describe_age(300) == "5 hours"
    assert main.describe_age(3 * 24 * 60) == "3 days"


# --- With the database ----------------------------------------------------------------------

RUN_ID = "pytest-stage6"


@pytest.fixture(scope="module")
def client():
    """A run dated 2000 so it never becomes the map's current run, removed afterwards."""
    issued = datetime(2000, 1, 2, tzinfo=UTC)
    valid_from, valid_to = risk.valid_window(date(2000, 1, 1))
    levels = {"G-N": 0.9, "A": 0.1}
    result = {
        "run": {
            "run_id": RUN_ID,
            "issued_at": issued,
            "data_time": issued - timedelta(hours=1),
            "rain_source": "observed",
            "model": "pytest",
            "is_test": True,
            "note": "pytest",
        },
        "risks": [
            WardRisk(
                ward_id=w,
                hazard=Hazard.FLOOD,
                valid_from=valid_from,
                valid_to=valid_to,
                probability=p,
                severity=risk.severity(p),
                trust=Trust.LOW,
                run_id=RUN_ID,
            )
            for w, p in levels.items()
        ],
    }
    try:
        risk.delete_run(RUN_ID)
        risk.save(result)
    except OperationalError:
        pytest.skip("database not running")
    yield TestClient(main.app)
    risk.delete_run(RUN_ID)


def test_health(client):
    assert client.get("/v1/health").json()["status"] == "ok"


def test_point_lookup_returns_the_ward_and_its_risk(client):
    body = client.get("/v1/risk", params={"lat": 19.0178, "lon": 72.8426, "run_id": RUN_ID})
    assert body.status_code == 200
    body = body.json()
    assert body["ward_id"] == "G-N"
    assert body["risks"][0]["severity"] == "red"
    assert body["risks"][0]["data_age_minutes"] > 60 * 24 * 365  # a run from 2000
    assert any(w.startswith("TEST data") for w in body["warnings"])
    assert "Flood risk RED: G-N." in body["warnings"]


def test_ward_endpoint_and_errors(client):
    assert (
        client.get("/v1/wards/A/risk", params={"run_id": RUN_ID}).json()["risks"][0]["severity"]
        == "green"
    )
    no_risk = client.get("/v1/wards/F-N/risk", params={"run_id": RUN_ID}).json()
    assert no_risk["risks"] == []
    assert client.get("/v1/wards/ZZ/risk").status_code == 404
    assert client.get("/v1/wards/A/risk", params={"run_id": "nope"}).status_code == 404
    assert client.get("/v1/risk", params={"lat": 18.5, "lon": 72.5}).status_code == 404
    assert client.get("/v1/risk", params={"lat": 95, "lon": 72.8}).status_code == 422


def test_map_geojson_has_every_ward(client):
    body = client.get("/v1/wards", params={"run_id": RUN_ID}).json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 24
    props = {f["properties"]["ward_id"]: f["properties"] for f in body["features"]}
    assert props["G-N"]["severity"] == "red"
    assert "severity" not in props["F-N"]  # no row in this run: drawn as no data
    assert body["run"]["run_id"] == RUN_ID
    runs = client.get("/v1/runs", params={"limit": 200}).json()
    assert RUN_ID in [r["run_id"] for r in runs]
