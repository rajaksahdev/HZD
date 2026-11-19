"""HazardLens API: ward risk from PostGIS for the map and other clients.

uvicorn api.main:app --reload            then open http://localhost:8000 (map) or /docs (API)

GET /v1/health                  database reachable?
GET /v1/runs?hazard=            recent risk runs, newest first
GET /v1/wards?hazard=&run_id=   GeoJSON of the 24 wards with their risk, for the map
GET /v1/wards/{ward_id}/risk    risk for one ward
GET /v1/risk?lat=&lon=          risk for the ward containing a point

hazard is flood (default) or heat. Without run_id the newest run of that hazard valid now is
used. Every answer carries the data age and the on-screen
warnings (TEST data, stale data, out-of-date validity, wards at orange or red).
"""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Connection, text
from sqlalchemy.exc import OperationalError

from api.db import engine
from api.schemas import SCHEMA_VERSION, Hazard, Severity, WardRisk

STALE_MINUTES = 180  # the pipeline runs hourly; three hours without new data means it stopped
WARN_LEVELS = (Severity.RED, Severity.ORANGE)
IST = timezone(timedelta(hours=5, minutes=30))
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
DISCLAIMER = (
    "Decision-support prototype. Official warnings come from IMD, NDMA and the Maharashtra "
    "state authorities."
)

app = FastAPI(title="HazardLens API", version=SCHEMA_VERSION, description=DISCLAIMER)


class RunInfo(BaseModel):
    run_id: str
    hazard: str | None = None
    issued_at: datetime
    data_time: datetime
    data_age_minutes: int
    rain_source: str
    model: str
    is_test: bool
    note: str | None
    valid_from: datetime | None
    valid_to: datetime | None


class RiskAnswer(BaseModel):
    ward_id: str
    name: str
    run: RunInfo | None
    risks: list[WardRisk]
    warnings: list[str]
    disclaimer: str = DISCLAIMER


def get_conn():
    try:
        with engine.connect() as conn:
            yield conn
    except OperationalError as exc:
        raise HTTPException(503, "database not reachable") from exc


Conn = Annotated[Connection, Depends(get_conn)]


# --- Pure helpers (tested without a database) ---------------------------------------------


def minutes_between(earlier: datetime, later: datetime) -> int:
    return max(0, int((later - earlier).total_seconds() // 60))


def describe_age(minutes: int) -> str:
    if minutes < 120:
        return f"{minutes} minutes"
    if minutes < 48 * 60:
        return f"{minutes // 60} hours"
    return f"{minutes // (24 * 60)} days"


RUN_COMMAND = {
    Hazard.FLOOD: "python -m models.risk --date today",
    Hazard.HEAT: "python -m models.heat --date today",
}
LOW_TRUST = {
    Hazard.FLOOD: (
        "Low trust: the flood labels are unreviewed and the model warns far more often "
        "than a heavy-rain rule. Treat levels as indicative."
    ),
}


def warnings_for(
    run: RunInfo | None, risks: list[WardRisk], now: datetime, hazard: Hazard = Hazard.FLOOD
) -> list[str]:
    """On-screen warnings, most important first."""
    hazard = Hazard(run.hazard) if run and run.hazard else hazard
    if run is None:
        command = RUN_COMMAND.get(hazard, "")
        return [f"No risk run for {hazard.value} in the database yet. Run: {command}"]
    out = []
    if run.is_test:
        out.append("TEST data: a hindcast or demo run, not a warning.")
    if run.valid_to and run.valid_to <= now:
        day = run.valid_from.astimezone(IST).strftime("%d %b %Y")
        out.append(f"Out of date: this risk was for {day}, not today.")
    elif run.valid_from and run.valid_from > now:
        day = run.valid_from.astimezone(IST).strftime("%d %b %Y")
        out.append(f"This risk is for {day}, a future day.")
    if run.data_age_minutes > STALE_MINUTES:
        out.append(f"Stale data: the newest input is {describe_age(run.data_age_minutes)} old.")
    for level in WARN_LEVELS:
        wards = sorted(r.ward_id for r in risks if r.severity == level)
        if wards:
            listed = ", ".join(wards) if len(wards) <= 8 else f"{len(wards)} wards"
            out.append(f"{hazard.value.capitalize()} risk {level.value.upper()}: {listed}.")
    degraded = [r.ward_id for r in risks if r.degraded]
    if degraded:
        out.append(f"Fallback rules used for {len(degraded)} wards.")
    if any(r.trust.value == "low" for r in risks):
        out.append(LOW_TRUST.get(hazard, "Low trust: treat levels as indicative."))
    return out


# --- Queries --------------------------------------------------------------------------------


def find_run(
    conn, run_id: str | None, now: datetime, hazard: Hazard = Hazard.FLOOD
) -> RunInfo | None:
    """The named run, or else the newest run of the hazard valid now, or else its newest run."""
    where = "WHERE r.run_id = :run_id" if run_id else "WHERE w.hazard = :hazard"
    row = (
        conn.execute(
            text(
                f"""
                SELECT r.*, min(w.valid_from) AS valid_from, max(w.valid_to) AS valid_to,
                       min(w.hazard) AS hazard
                FROM risk_runs r LEFT JOIN ward_risk w ON w.run_id = r.run_id
                {where}
                GROUP BY r.run_id
                ORDER BY coalesce(min(w.valid_from) <= :now AND :now < max(w.valid_to), false)
                         DESC, r.issued_at DESC
                LIMIT 1
                """
            ),
            {"run_id": run_id, "now": now, "hazard": hazard.value},
        )
        .mappings()
        .first()
    )
    if row is None:
        if run_id:
            raise HTTPException(404, f"no run {run_id}")
        return None
    return RunInfo(**row, data_age_minutes=minutes_between(row["data_time"], now))


def ward_risks(conn, run: RunInfo | None, ward_id: str | None = None) -> list[WardRisk]:
    if run is None:
        return []
    rows = (
        conn.execute(
            text(
                "SELECT ward_id, hazard, valid_from, valid_to, probability, severity, trust, "
                "run_id, degraded, value, unit FROM ward_risk WHERE run_id = :run_id "
                "AND (CAST(:ward_id AS text) IS NULL OR ward_id = :ward_id) ORDER BY ward_id"
            ),
            {"run_id": run.run_id, "ward_id": ward_id},
        )
        .mappings()
        .all()
    )
    return [WardRisk(**row, data_age_minutes=run.data_age_minutes) for row in rows]


def answer(conn, ward: dict, run_id: str | None, hazard: Hazard) -> RiskAnswer:
    now = datetime.now(UTC)
    run = find_run(conn, run_id, now, hazard)
    risks = ward_risks(conn, run, ward["ward_id"])
    return RiskAnswer(
        ward_id=ward["ward_id"],
        name=ward["name"],
        run=run,
        risks=risks,
        warnings=warnings_for(run, risks, now, hazard),
    )


# --- Routes ---------------------------------------------------------------------------------


@app.get("/v1/health")
def health(conn: Conn):
    conn.execute(text("SELECT 1"))
    return {"status": "ok", "schema_version": SCHEMA_VERSION}


@app.get("/v1/runs")
def runs(conn: Conn, limit: int = Query(20, ge=1, le=200), hazard: Hazard | None = None):
    rows = conn.execute(
        text(
            """
            SELECT r.run_id, r.issued_at, r.rain_source, r.is_test,
                   min(w.valid_from) AS valid_from, count(w.id) AS wards,
                   min(w.hazard) AS hazard
            FROM risk_runs r LEFT JOIN ward_risk w ON w.run_id = r.run_id
            GROUP BY r.run_id
            HAVING CAST(:hazard AS text) IS NULL OR min(w.hazard) = :hazard
            ORDER BY r.issued_at DESC LIMIT :limit
            """
        ),
        {"limit": limit, "hazard": hazard.value if hazard else None},
    ).mappings()
    return [dict(r) for r in rows]


@app.get("/v1/wards")
def wards_geojson(conn: Conn, run_id: str | None = None, hazard: Hazard = Hazard.FLOOD):
    now = datetime.now(UTC)
    run = find_run(conn, run_id, now, hazard)
    risks = {r.ward_id: r for r in ward_risks(conn, run)}
    shapes = conn.execute(
        text(
            "SELECT ward_id, name, "
            "ST_AsGeoJSON(ST_SimplifyPreserveTopology(geometry, 0.0002), 5) AS geom "
            "FROM wards ORDER BY ward_id"
        )
    ).mappings()
    features = []
    for shape in shapes:
        risk = risks.get(shape["ward_id"])
        props = {"ward_id": shape["ward_id"], "name": shape["name"]}
        if risk:
            props.update(
                probability=risk.probability,
                severity=risk.severity.value,
                trust=risk.trust.value,
                degraded=risk.degraded,
                value=risk.value,
                unit=risk.unit,
            )
        features.append(
            {"type": "Feature", "geometry": json.loads(shape["geom"]), "properties": props}
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "hazard": run.hazard if run and run.hazard else hazard.value,
        "run": run.model_dump(mode="json") if run else None,
        "warnings": warnings_for(run, list(risks.values()), now, hazard),
        "disclaimer": DISCLAIMER,
    }


@app.get("/v1/wards/{ward_id}/risk", response_model=RiskAnswer)
def ward_risk(conn: Conn, ward_id: str, run_id: str | None = None, hazard: Hazard = Hazard.FLOOD):
    ward = (
        conn.execute(text("SELECT ward_id, name FROM wards WHERE ward_id = :w"), {"w": ward_id})
        .mappings()
        .first()
    )
    if ward is None:
        raise HTTPException(404, f"unknown ward {ward_id}; ids look like A, F-N, K-W")
    return answer(conn, dict(ward), run_id, hazard)


@app.get("/v1/risk", response_model=RiskAnswer)
def point_risk(
    conn: Conn,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    run_id: str | None = None,
    hazard: Hazard = Hazard.FLOOD,
):
    ward = (
        conn.execute(
            text(
                "SELECT ward_id, name FROM wards "
                "WHERE ST_Contains(geometry, ST_SetSRID(ST_Point(:lon, :lat), 4326)) LIMIT 1"
            ),
            {"lat": lat, "lon": lon},
        )
        .mappings()
        .first()
    )
    if ward is None:
        raise HTTPException(404, "the point is outside the 24 pilot wards of Mumbai")
    return answer(conn, dict(ward), run_id, hazard)


if WEB_DIST.exists():  # the built map (cd web && npm run build); mounted last so /v1 wins
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
