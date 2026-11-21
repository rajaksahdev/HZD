"""HazardLens API: ward risk from PostGIS for the map and other clients.

uvicorn api.main:app --reload            then open http://localhost:8000 (map) or /docs (API)

GET /v1/health                  database reachable?
GET /v1/runs?hazard=            recent risk runs, newest first
GET /v1/wards?hazard=&run_id=   GeoJSON of the 24 wards with their risk, for the map
GET /v1/wards/{ward_id}/risk    risk for one ward
GET /v1/risk?lat=&lon=          risk for the ward containing a point
GET /v1/impact?hazard=&run_id=  people at risk, impact score and nearest shelter per ward
GET /v1/wards/{ward_id}/impact  the same for one ward, with its routes as GeoJSON
GET /v1/facilities?kind=        hospitals and shelter candidates as GeoJSON
GET /v1/alerts?hazard=&ward_id=&at=   signed alerts active now (or at a time)
GET /v1/alerts/{alert_id}       every signed message of one alert, in sequence
GET /v1/alerts/{alert_id}/cap   the latest (or ?sequence=) message as CAP 1.2 XML
GET /v1/alert-keys              public keys to check alert signatures (Ed25519)

Alerts are only read here: the engine that signs them (alerts/engine.py) runs elsewhere with the
private key, which this process never opens.

hazard is flood (default) or heat. Without run_id the newest run of that hazard valid now is
used. Every answer carries the data age and the on-screen
warnings (TEST data, stale data, out-of-date validity, wards at orange or red).
"""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Connection, text
from sqlalchemy.exc import OperationalError

from alerts.engine import cap_xml, messages
from api.db import engine
from api.schemas import (
    SCHEMA_VERSION,
    Alert,
    Hazard,
    Route,
    Severity,
    WardImpact,
    WardRisk,
)

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
CAVEAT = {  # shown with every run of the hazard, whatever the forecast trust
    Hazard.FLOOD: (
        "Provisional: the flood labels are unreviewed and the model warns far more often "
        "than a heavy-rain rule. Treat levels as indicative."
    ),
}
LOW_TRUST = {  # Stage 11: trust is how much the forecast models disagree
    Hazard.FLOOD: "the rain forecast models disagree",
    Hazard.HEAT: "the temperature forecast models disagree",
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
    if hazard in CAVEAT and any(r.severity != Severity.GREEN for r in risks):
        out.append(CAVEAT[hazard])
    low = sorted(r.ward_id for r in risks if r.trust.value == "low")
    if low:
        listed = ", ".join(low) if len(low) <= 8 else f"{len(low)} wards"
        reason = LOW_TRUST.get(hazard, "the inputs are uncertain")
        out.append(f"Low trust in {listed}: {reason}. Treat levels there as indicative.")
    return out


def impact_rows(
    hazard: Hazard,
    run: RunInfo | None,
    risks: list[WardRisk],
    vulnerability: dict[str, dict],
    routes: dict[tuple[str, str], Route],
) -> list[WardImpact]:
    """Stage 12: people at risk = probability x people exposed (flood: on low-lying land; heat:
    everyone); impact score = people at risk x (0.5 + vulnerability index)."""
    by_ward = {r.ward_id: r for r in risks}
    out = []
    for ward_id in sorted(set(vulnerability) | set(by_ward)):
        v, r = vulnerability.get(ward_id), by_ward.get(ward_id)
        exposed = None
        if v is not None:
            exposed = v["low_lying_pop"] if hazard == Hazard.FLOOD else v["population"]
        people = score = None
        if r is not None and exposed is not None:
            people = r.probability * exposed
            score = people * (0.5 + v["vulnerability_index"])
        out.append(
            WardImpact(
                ward_id=ward_id,
                hazard=hazard,
                run_id=run.run_id if run else None,
                population=v["population"] if v else None,
                exposed=exposed,
                vulnerability_index=v["vulnerability_index"] if v else None,
                probability=r.probability if r else None,
                severity=r.severity if r else None,
                people_at_risk=None if people is None else round(people),
                impact_score=None if score is None else round(score),
                shelter=routes.get((ward_id, "shelter")),
                hospital=routes.get((ward_id, "hospital")),
            )
        )
    return out


def route_warnings(
    hazard: Hazard, run: RunInfo | None, risks: list[WardRisk], routes_run: str | None
) -> list[str]:
    """Say when the routes shown do not avoid the roads this run puts at risk."""
    if routes_run is None:
        return ["No routes yet. Run: python -m models.impact --build"]
    hot = any(r.severity in WARN_LEVELS for r in risks)
    if hazard == Hazard.FLOOD and run and hot and routes_run != run.run_id:
        return [
            "Routes shown do not avoid roads likely to flood in this run. "
            f"Run: python -m models.impact --routes {run.run_id}"
        ]
    return []


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


def has_table(conn, name: str) -> bool:
    return conn.execute(text("SELECT to_regclass(:n) IS NOT NULL"), {"n": name}).scalar()


def vulnerability_by_ward(conn) -> dict[str, dict]:
    if not has_table(conn, "ward_vulnerability"):
        return {}
    rows = conn.execute(text("SELECT * FROM ward_vulnerability")).mappings()
    return {r["ward_id"]: dict(r) for r in rows}


ROUTE_FIELDS = (
    "ward_id, kind, run_id, status, facility_id, facility_name, distance_m, baseline_m, "
    "avoided_km, people_cut_off"
)


def routes_for(
    conn, run: RunInfo | None, ward_id: str | None = None, geometry: bool = False
) -> tuple[str | None, dict[tuple[str, str], Route], dict]:
    """Routes for the run, or else the baseline routes (which avoid nothing). Returns the run id
    the routes belong to, the routes by (ward, kind) and, if asked, their GeoJSON lines."""
    if not has_table(conn, "ward_route"):
        return None, {}, {}
    geom = ", ST_AsGeoJSON(geometry, 6) AS geom" if geometry else ""
    for run_id in ([run.run_id] if run else []) + ["baseline"]:
        rows = (
            conn.execute(
                text(
                    f"SELECT {ROUTE_FIELDS}{geom} FROM ward_route WHERE run_id = :run_id "
                    "AND (CAST(:ward_id AS text) IS NULL OR ward_id = :ward_id)"
                ),
                {"run_id": run_id, "ward_id": ward_id},
            )
            .mappings()
            .all()
        )
        if rows:
            routes = {
                (r["ward_id"], r["kind"]): Route(**{k: v for k, v in r.items() if k != "geom"})
                for r in rows
            }
            lines = {(r["ward_id"], r["kind"]): r.get("geom") for r in rows}
            return run_id, routes, lines
    return None, {}, {}


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
    impacts = {
        i.ward_id: i
        for i in impact_rows(hazard, run, list(risks.values()), vulnerability_by_ward(conn), {})
    }
    features = []
    for shape in shapes:
        risk = risks.get(shape["ward_id"])
        props = {"ward_id": shape["ward_id"], "name": shape["name"]}
        found = impacts.get(shape["ward_id"])
        if found and found.population is not None:
            props.update(
                population=found.population,
                vulnerability_index=found.vulnerability_index,
                people_at_risk=found.people_at_risk,
                impact_score=found.impact_score,
            )
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


class ImpactAnswer(BaseModel):
    hazard: Hazard
    run: RunInfo | None
    routes_run_id: str | None  # the run whose roads the routes avoid, or "baseline"
    wards: list[WardImpact]
    warnings: list[str]
    disclaimer: str = DISCLAIMER


def impact_answer(conn, run_id: str | None, hazard: Hazard, ward_id: str | None = None):
    now = datetime.now(UTC)
    run = find_run(conn, run_id, now, hazard)
    hazard = Hazard(run.hazard) if run and run.hazard else hazard
    risks = ward_risks(conn, run, ward_id)
    routes_run, routes, lines = routes_for(conn, run, ward_id, geometry=ward_id is not None)
    vulnerability = vulnerability_by_ward(conn)
    if ward_id:
        vulnerability = {k: v for k, v in vulnerability.items() if k == ward_id}
    wards = impact_rows(hazard, run, risks, vulnerability, routes)
    warnings = route_warnings(hazard, run, risks, routes_run)
    if not vulnerability:
        warnings.insert(0, "No population data yet. Run: python -m models.impact --build")
    body = ImpactAnswer(
        hazard=hazard, run=run, routes_run_id=routes_run, wards=wards, warnings=warnings
    )
    return body, lines


@app.get("/v1/impact", response_model=ImpactAnswer)
def impact(conn: Conn, run_id: str | None = None, hazard: Hazard = Hazard.FLOOD):
    return impact_answer(conn, run_id, hazard)[0]


@app.get("/v1/wards/{ward_id}/impact")
def ward_impact(conn: Conn, ward_id: str, run_id: str | None = None, hazard: Hazard = Hazard.FLOOD):
    known = conn.execute(text("SELECT 1 FROM wards WHERE ward_id = :w"), {"w": ward_id}).first()
    if known is None:
        raise HTTPException(404, f"unknown ward {ward_id}; ids look like A, F-N, K-W")
    body, lines = impact_answer(conn, run_id, hazard, ward_id)
    features = []
    for ward in body.wards:
        for route in (ward.shelter, ward.hospital):
            line = lines.get((route.ward_id, route.kind)) if route else None
            if line:
                props = route.model_dump(mode="json", exclude={"schema_version"})
                features.append(
                    {"type": "Feature", "geometry": json.loads(line), "properties": props}
                )
    out = body.model_dump(mode="json")
    out["routes"] = {"type": "FeatureCollection", "features": features}
    return out


@app.get("/v1/facilities")
def facilities(conn: Conn, kind: str | None = None, ward_id: str | None = None):
    """kind: hospital, shelter (every shelter candidate), or a single kind such as school."""
    if not has_table(conn, "facilities"):
        return {"type": "FeatureCollection", "features": []}
    rows = conn.execute(
        text(
            "SELECT facility_id, kind, name, ward_id, emergency, official, "
            "ST_X(geometry) AS lon, ST_Y(geometry) AS lat FROM facilities "
            "WHERE (CAST(:kind AS text) IS NULL OR kind = :kind "
            "  OR (:kind = 'shelter' AND kind <> 'hospital')) "
            "AND (CAST(:ward_id AS text) IS NULL OR ward_id = :ward_id) ORDER BY facility_id"
        ),
        {"kind": kind, "ward_id": ward_id},
    ).mappings()
    features = []
    for r in rows:
        props = {k: r[k] for k in r if k not in ("lon", "lat")}
        props["group"] = "hospital" if r["kind"] == "hospital" else "shelter"
        point = {"type": "Point", "coordinates": [round(r["lon"], 6), round(r["lat"], 6)]}
        features.append({"type": "Feature", "geometry": point, "properties": props})
    return {
        "type": "FeatureCollection",
        "features": features,
        "note": "Shelter candidates are schools, community centres and assembly points from "
        "OpenStreetMap, not designated shelters.",
    }


class AlertsAnswer(BaseModel):
    at: datetime
    alerts: list[Alert]
    warnings: list[str]
    disclaimer: str = DISCLAIMER


def alert_warnings(alerts: list[Alert]) -> list[str]:
    out = []
    if any(a.is_test for a in alerts):
        out.append("TEST alerts: produced by a prototype from demo or hindcast runs.")
    red = sorted(a.ward_id for a in alerts if a.severity == Severity.RED)
    if red:
        out.append(f"Red alerts: {', '.join(red)}.")
    return out


@app.get("/v1/alerts", response_model=AlertsAnswer)
def alerts_active(
    conn: Conn,
    hazard: Hazard | None = None,
    ward_id: str | None = None,
    at: datetime | None = None,
):
    """Alerts in force at `at` (default now): the latest message of each alert sent by then,
    unless it is a cancel or has expired."""
    at = at or datetime.now(UTC)
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    if not has_table(conn, "alert_messages"):
        return AlertsAnswer(at=at, alerts=[], warnings=[])
    rows = conn.execute(
        text(
            """
            SELECT DISTINCT ON (alert_id) body, message_type, expires_at
            FROM alert_messages
            WHERE status = 'issued' AND issued_at <= :at
              AND (CAST(:hazard AS text) IS NULL OR hazard = :hazard)
              AND (CAST(:ward_id AS text) IS NULL OR ward_id = :ward_id)
            ORDER BY alert_id, sequence DESC
            """
        ),
        {"at": at, "hazard": hazard.value if hazard else None, "ward_id": ward_id},
    ).mappings()
    alerts = [
        Alert(**json.loads(r["body"]))
        for r in rows
        if r["message_type"] != "cancel" and r["expires_at"] > at
    ]
    alerts.sort(key=lambda a: (a.hazard, a.ward_id))
    return AlertsAnswer(at=at, alerts=alerts, warnings=alert_warnings(alerts))


@app.get("/v1/alert-keys")
def alert_keys(conn: Conn):
    if not has_table(conn, "alert_keys"):
        return []
    rows = conn.execute(
        text(
            "SELECT key_id, public_key, created_at, retired_at FROM alert_keys ORDER BY created_at"
        )
    ).mappings()
    return [dict(r, algorithm="Ed25519") for r in rows]


@app.get("/v1/alerts/{alert_id}")
def alert_history(conn: Conn, alert_id: str):
    found = messages(conn, alert_id) if has_table(conn, "alert_messages") else []
    if not found:
        raise HTTPException(404, f"no alert {alert_id}")
    return {"alert_id": alert_id, "messages": found, "disclaimer": DISCLAIMER}


@app.get("/v1/alerts/{alert_id}/cap")
def alert_cap(conn: Conn, alert_id: str, sequence: int | None = Query(None, ge=1)):
    if not has_table(conn, "alert_messages"):
        raise HTTPException(404, f"no alert {alert_id}")
    try:
        xml = cap_xml(conn, alert_id, sequence)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(xml, media_type="application/xml")


if WEB_DIST.exists():  # the built map (cd web && npm run build); mounted last so /v1 wins
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
