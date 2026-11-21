"""Alert engine (Stage 13): from a risk run to signed alert, update and cancel messages.

Severity rules (per hazard, per ward):
- A ward at orange or red in a run gets an alert; its expiry is the end of the risk window.
- Next run: a changed level, trust or a later window sends an update (same alert_id, next
  sequence); a ward back to yellow or green sends a cancel; nothing new sends nothing.
- Sent at once: orange alerts and their updates. Held for an operator (two-factor login):
  anything at red, anything that lowers or ends a red alert, a new alert with low trust (the
  forecast models disagree), and a new alert in a ward over its false-alarm target.
- A draft still waiting when a newer run arrives is superseded by that run's decision.
- Every message is TEST unless the run is live and ALERTS_LIVE=1 is set.

Each message is signed with Ed25519 (alerts/keys.py) when it is sent, never before, and stored
with its exact JSON body. Operators can also cancel any active alert by hand (a false alarm).

False-alarm target: per ward and hazard, the largest share of alerts that no flood report
confirms (default 0.5). `--targets` measures it on the 2024 and 2025 monsoon replays: every
day-ahead run (Stage 7 archive) where the ward reached orange counts as an alert-day.

python -m alerts.engine --run RUN_ID|latest [--hazard flood] [--as-issued]
python -m alerts.engine --pending
python -m alerts.engine --approve MESSAGE_ID --operator NAME
python -m alerts.engine --reject MESSAGE_ID --operator NAME --reason TEXT
python -m alerts.engine --cancel ALERT_ID --operator NAME --reason TEXT
python -m alerts.engine --cap ALERT_ID [--sequence N] [--out FILE]
python -m alerts.engine --targets [--report docs/alerts.md]
python -m alerts.engine --set-target WARD HAZARD SHARE
"""

import argparse
import getpass
import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from alerts import cap, keys, operators
from api.db import AlertMessage, AlertTarget, RiskRun, SessionLocal, WardRiskRow, init_db
from api.schemas import Alert, Hazard, MessageType, Severity, Trust, WardRisk

IST = timezone(timedelta(hours=5, minutes=30))
RANK = {Severity.GREEN: 0, Severity.YELLOW: 1, Severity.ORANGE: 2, Severity.RED: 3}


@dataclass(frozen=True)
class Rule:
    alert_from: Severity  # this level or above gets an alert
    approval_from: Severity  # this level or above waits for an operator


RULES = {
    Hazard.FLOOD: Rule(Severity.ORANGE, Severity.RED),
    Hazard.HEAT: Rule(Severity.ORANGE, Severity.RED),
}
ACTION = {
    (Hazard.FLOOD, Severity.ORANGE): "FLOOD_AVOID_LOW_ROADS",
    (Hazard.FLOOD, Severity.RED): "FLOOD_MOVE_UP",
    (Hazard.HEAT, Severity.ORANGE): "HEAT_LIMIT_SUN",
    (Hazard.HEAT, Severity.RED): "HEAT_STAY_COOL",
}
ALL_CLEAR = "ALL_CLEAR"
CANCEL_LIFETIME = timedelta(hours=6)  # a cancel stays visible this long after an alert expired
DEFAULT_MAX_FALSE_ALARM = 0.5
MIN_ALERT_DAYS = 5  # fewer measured alert-days than this: the target is not judged
TARGET_YEARS = (2024, 2025)
TARGET_LEAD = 1  # day-ahead runs, the lead an alert would normally be sent at


def live() -> bool:
    return os.getenv("ALERTS_LIVE") == "1"


def at_least(level: Severity, floor: Severity) -> bool:
    return RANK[level] >= RANK[floor]


# --- Decisions (pure) -----------------------------------------------------------------------


@dataclass
class Draft:
    """A message the rules want to send. `reasons` empty: send now; else hold for an operator."""

    ward_id: str
    hazard: Hazard
    message_type: MessageType
    severity: Severity
    trust: Trust
    action_code: str
    onset: datetime | None
    expires_at: datetime
    run_id: str | None
    alert_id: str | None = None  # None for a new alert
    reasons: list[str] = field(default_factory=list)


def new_alert_id(hazard: Hazard, ward_id: str, onset: datetime) -> str:
    return f"{hazard.value}-{ward_id}-{onset.astimezone(IST):%Y%m%d}-{secrets.token_hex(3)}"


def decide(
    risks: list[WardRisk],
    active: dict[str, Alert],
    over_target: set[str],
    now: datetime,
) -> list[Draft]:
    """What to send for one run. `active`: the current alert per ward for this hazard."""
    drafts = []
    for r in risks:
        rule = RULES.get(r.hazard)
        if rule is None:
            continue
        current = active.get(r.ward_id)
        wanted = at_least(r.severity, rule.alert_from) and r.valid_to > now
        red_now = current is not None and at_least(current.severity, rule.approval_from)
        if wanted:
            if current is not None and (
                current.severity == r.severity
                and current.trust == r.trust
                and current.expires_at >= r.valid_to
            ):
                continue  # nothing new
            draft = Draft(
                ward_id=r.ward_id,
                hazard=r.hazard,
                message_type=MessageType.ALERT if current is None else MessageType.UPDATE,
                severity=r.severity,
                trust=r.trust,
                action_code=ACTION[(r.hazard, r.severity)],
                onset=r.valid_from,
                expires_at=r.valid_to,
                run_id=r.run_id,
                alert_id=current.alert_id if current else None,
            )
            if at_least(r.severity, rule.approval_from):
                draft.reasons.append(f"{r.severity.value} alert")
            elif red_now:
                draft.reasons.append(f"lowers a {current.severity.value} alert")
            if current is None and r.trust == Trust.LOW:
                draft.reasons.append("low trust: the forecast models disagree")
            if current is None and r.ward_id in over_target:
                draft.reasons.append("ward over its false-alarm target")
            drafts.append(draft)
        elif current is not None:
            draft = Draft(
                ward_id=r.ward_id,
                hazard=r.hazard,
                message_type=MessageType.CANCEL,
                severity=current.severity,
                trust=r.trust,
                action_code=ALL_CLEAR,
                onset=current.onset,
                expires_at=max(current.expires_at, now + CANCEL_LIFETIME),
                run_id=r.run_id,
                alert_id=current.alert_id,
                reasons=[f"ends a {current.severity.value} alert"] if red_now else [],
            )
            drafts.append(draft)
    return drafts


# --- Storage --------------------------------------------------------------------------------


def _second(when: datetime) -> datetime:
    return when.astimezone(UTC).replace(microsecond=0)


def latest_issued(session: Session, alert_id: str) -> AlertMessage | None:
    return session.scalars(
        select(AlertMessage)
        .where(AlertMessage.alert_id == alert_id, AlertMessage.status == "issued")
        .order_by(AlertMessage.sequence.desc())
        .limit(1)
    ).first()


def is_active(message: AlertMessage | None, now: datetime) -> bool:
    return (
        message is not None
        and message.message_type != MessageType.CANCEL
        and message.expires_at > now
    )


def active_alerts(session: Session, hazard: Hazard, now: datetime) -> dict[str, Alert]:
    """The current (latest, not cancelled, not expired) alert per ward."""
    latest = (
        select(AlertMessage.alert_id, func.max(AlertMessage.sequence).label("seq"))
        .where(
            AlertMessage.hazard == hazard.value,
            AlertMessage.status == "issued",
            AlertMessage.issued_at <= now,
        )
        .group_by(AlertMessage.alert_id)
        .subquery()
    )
    rows = session.scalars(
        select(AlertMessage)
        .join(
            latest,
            (AlertMessage.alert_id == latest.c.alert_id) & (AlertMessage.sequence == latest.c.seq),
        )
        .where(AlertMessage.status == "issued")
        .order_by(AlertMessage.issued_at)
    )
    return {m.ward_id: Alert(**json.loads(m.body)) for m in rows if is_active(m, now)}


def over_target(session: Session, hazard: Hazard) -> set[str]:
    rows = session.scalars(select(AlertTarget).where(AlertTarget.hazard == hazard.value))
    return {
        t.ward_id
        for t in rows
        if t.false_alarm is not None
        and (t.alerts or 0) >= MIN_ALERT_DAYS
        and t.false_alarm > t.max_false_alarm
    }


def store(session: Session, draft: Draft, is_test: bool, now: datetime) -> AlertMessage:
    row = AlertMessage(
        alert_id=draft.alert_id or new_alert_id(draft.hazard, draft.ward_id, draft.onset or now),
        sequence=0,  # set when issued
        message_type=draft.message_type.value,
        hazard=draft.hazard.value,
        ward_id=draft.ward_id,
        severity=draft.severity.value,
        trust=draft.trust.value,
        action_code=draft.action_code,
        onset=draft.onset,
        expires_at=draft.expires_at,
        is_test=is_test,
        run_id=draft.run_id,
        status="pending",
        reason="; ".join(draft.reasons)[:300] or None,
        created_at=now,
    )
    session.add(row)
    session.flush()
    return row


def issue(
    session: Session,
    row: AlertMessage,
    private: Ed25519PrivateKey,
    now: datetime,
    approved_by: str | None = None,
) -> Alert:
    """Sign and send a stored draft: next sequence for its alert, signature over the JSON."""
    previous = latest_issued(session, row.alert_id)
    if row.message_type != MessageType.ALERT and not is_active(previous, now):
        row.status, row.closed_at = "withdrawn", now
        raise ValueError(f"alert {row.alert_id} is no longer active")
    issued_at = _second(now)
    if row.expires_at <= issued_at:
        row.status, row.closed_at = "expired", now
        raise ValueError(f"message {row.id} expired before it was sent")
    alert = Alert(
        alert_id=row.alert_id,
        key_id=keys.key_id(keys.public_raw(private)),
        sequence=(previous.sequence if previous else 0) + 1,
        message_type=MessageType(row.message_type),
        hazard=Hazard(row.hazard),
        ward_id=row.ward_id,
        severity=Severity(row.severity),
        issued_at=issued_at,
        expires_at=_second(row.expires_at),
        trust=Trust(row.trust),
        action_code=row.action_code,
        is_test=row.is_test,
        onset=_second(row.onset) if row.onset else None,
        run_id=row.run_id,
    )
    body = alert.model_dump(mode="json")
    body["signature"] = keys.sign(private, body)
    row.sequence, row.issued_at, row.status = alert.sequence, issued_at, "issued"
    row.closed_at = issued_at
    row.key_id, row.signature, row.body = alert.key_id, body["signature"], json.dumps(body)
    row.approved_by = approved_by
    session.flush()
    return Alert(**body)


# --- Runs -----------------------------------------------------------------------------------


def run_risks(session: Session, run_id: str) -> tuple[RiskRun, list[WardRisk]]:
    run = session.get(RiskRun, run_id)
    if run is None:
        raise ValueError(f"no run {run_id}")
    rows = session.scalars(
        select(WardRiskRow).where(WardRiskRow.run_id == run_id).order_by(WardRiskRow.ward_id)
    )
    risks = [
        WardRisk(
            ward_id=r.ward_id,
            hazard=r.hazard,
            valid_from=r.valid_from,
            valid_to=r.valid_to,
            probability=r.probability,
            severity=r.severity,
            trust=r.trust,
            run_id=r.run_id,
            degraded=r.degraded,
            value=r.value,
            unit=r.unit,
        )
        for r in rows
    ]
    return run, risks


def latest_run(session: Session, hazard: Hazard) -> str | None:
    return session.execute(
        text(
            "SELECT r.run_id FROM risk_runs r JOIN ward_risk w ON w.run_id = r.run_id "
            "WHERE w.hazard = :h ORDER BY r.issued_at DESC LIMIT 1"
        ),
        {"h": hazard.value},
    ).scalar()


def process_run(
    run_id: str,
    now: datetime | None = None,
    private: Ed25519PrivateKey | None = None,
) -> dict:
    """Apply the rules to one run: send what may be sent, hold the rest for an operator."""
    init_db()
    private = private or keys.load()
    keys.register(private)
    with SessionLocal.begin() as session:
        run, risks = run_risks(session, run_id)
        now = now or datetime.now(UTC)
        hazards = {r.hazard for r in risks}
        if len(hazards) != 1:
            raise ValueError(f"run {run_id} has {len(hazards)} hazards, expected one")
        hazard = hazards.pop()
        wards = [r.ward_id for r in risks]
        # Drafts still waiting from earlier runs give way to this run's decision; expired ones
        # are left alone (they lapse by their expiry). A never-sent alert keeps its alert_id.
        waiting = session.execute(
            text(
                "UPDATE alert_messages SET status = 'superseded', closed_at = :now "
                "WHERE status = 'pending' AND hazard = :h AND ward_id = ANY(:w) "
                "AND created_at <= :now AND expires_at > :now "
                "RETURNING ward_id, alert_id, message_type, status"
            ),
            {"h": hazard.value, "w": wards, "now": now},
        ).all()
        superseded = sum(w.status == "superseded" for w in waiting)
        unsent = {
            w.ward_id: w.alert_id
            for w in waiting
            if w.status == "superseded" and w.message_type == MessageType.ALERT
        }
        drafts = decide(
            risks, active_alerts(session, hazard, now), over_target(session, hazard), now
        )
        for draft in drafts:
            if draft.message_type == MessageType.ALERT and draft.ward_id in unsent:
                draft.alert_id = unsent[draft.ward_id]
        is_test = run.is_test or not live()
        out = {"run_id": run_id, "hazard": hazard.value, "superseded": superseded}
        out |= {"issued": [], "pending": []}
        for draft in drafts:
            row = store(session, draft, is_test, now)
            if draft.reasons:
                out["pending"].append((row.id, row.alert_id, row.message_type, row.reason))
            else:
                alert = issue(session, row, private, now)
                out["issued"].append((alert.alert_id, alert.sequence, alert.message_type.value))
    return out


# --- Operator actions -----------------------------------------------------------------------


def _login(session: Session, name: str, password: str, code: str, now: datetime):
    try:
        return operators.login(session, name, password, code, now)
    except operators.LoginError:
        session.commit()  # keep the failure count
        raise


def approve(
    message_id: int,
    name: str,
    password: str,
    code: str,
    now: datetime | None = None,
    private: Ed25519PrivateKey | None = None,
) -> Alert:
    now = now or datetime.now(UTC)
    private = private or keys.load()
    keys.register(private)
    with SessionLocal() as session:
        op = _login(session, name, password, code, now)
        row = session.get(AlertMessage, message_id, with_for_update=True)
        if row is None or row.status != "pending":
            session.commit()
            raise ValueError(f"message {message_id} is not waiting for approval")
        try:
            alert = issue(session, row, private, now, approved_by=op.name)
        except ValueError:
            session.commit()  # keep the withdrawn or expired status
            raise
        session.commit()
        return alert


def reject(message_id: int, name: str, password: str, code: str, reason: str, now=None) -> None:
    now = now or datetime.now(UTC)
    with SessionLocal() as session:
        op = _login(session, name, password, code, now)
        row = session.get(AlertMessage, message_id, with_for_update=True)
        if row is None or row.status != "pending":
            session.commit()
            raise ValueError(f"message {message_id} is not waiting for approval")
        row.status, row.approved_by, row.closed_at = "rejected", op.name, now
        row.reason = f"{row.reason or ''}; rejected: {reason}".strip("; ")[:300]
        session.commit()


def cancel(
    alert_id: str,
    name: str,
    password: str,
    code: str,
    reason: str,
    now: datetime | None = None,
    private: Ed25519PrivateKey | None = None,
) -> Alert:
    """An operator ends an active alert by hand, e.g. a false alarm."""
    now = now or datetime.now(UTC)
    private = private or keys.load()
    keys.register(private)
    with SessionLocal() as session:
        op = _login(session, name, password, code, now)
        current = latest_issued(session, alert_id)
        if not is_active(current, now):
            session.commit()
            raise ValueError(f"alert {alert_id} is not active")
        session.execute(
            text(
                "UPDATE alert_messages SET status = 'superseded', closed_at = :now "
                "WHERE alert_id = :a AND status = 'pending'"
            ),
            {"a": alert_id, "now": now},
        )
        draft = Draft(
            ward_id=current.ward_id,
            hazard=Hazard(current.hazard),
            message_type=MessageType.CANCEL,
            severity=Severity(current.severity),
            trust=Trust(current.trust),
            action_code=ALL_CLEAR,
            onset=current.onset,
            expires_at=max(current.expires_at, now + CANCEL_LIFETIME),
            run_id=current.run_id,
            alert_id=alert_id,
            reasons=[f"cancelled by operator: {reason}"],
        )
        row = store(session, draft, current.is_test, now)
        alert = issue(session, row, private, now, approved_by=op.name)
        session.commit()
        return alert


# --- Reading back ---------------------------------------------------------------------------


def messages(session_or_conn, alert_id: str) -> list[Alert]:
    rows = session_or_conn.execute(
        text(
            "SELECT body FROM alert_messages WHERE alert_id = :a AND status = 'issued' "
            "ORDER BY sequence"
        ),
        {"a": alert_id},
    ).scalars()
    return [Alert(**json.loads(b)) for b in rows]


def ward_rings(conn, ward_id: str, tolerance: float = 0.001) -> tuple[str, list]:
    """The ward's official name and its simplified outer rings as (lon, lat), for CAP."""
    row = conn.execute(
        text(
            "SELECT name, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geometry, :t), 5) AS g "
            "FROM wards WHERE ward_id = :w"
        ),
        {"w": ward_id, "t": tolerance},
    ).first()
    if row is None:
        return ward_id, []
    geom = json.loads(row.g)
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    rings = [[tuple(p) for p in poly[0]] for poly in polys if len(poly[0]) >= 4]
    return row.name, rings


def cap_xml(conn, alert_id: str, sequence: int | None = None) -> str:
    history = messages(conn, alert_id)
    if not history:
        raise ValueError(f"no issued messages for alert {alert_id}")
    chosen = [m for m in history if sequence is None or m.sequence == sequence]
    if not chosen:
        raise ValueError(f"alert {alert_id} has no message {sequence}")
    message = chosen[-1]
    name, rings = ward_rings(conn, message.ward_id)
    return cap.to_cap(message, name, rings, history)


# --- False-alarm targets --------------------------------------------------------------------


def season_alert_days(years=TARGET_YEARS, lead: int = TARGET_LEAD) -> pd.DataFrame:
    """Every monsoon day of the years as the day-ahead run saw it: one row per ward-day with
    the flood probability, whether it reached the alert level, and the heavy-rain rule."""
    from models import flood, replay, risk, tide

    model, meta = flood.load_final()
    tide_model = tide.load_model()
    wards = flood.ward_points().index
    rows = []
    for year in years:
        start, end = date(year, 6, 1), date(year, 9, 30)
        for n in range((end - start).days + 1):
            day = start + timedelta(days=n)
            grid = pd.DataFrame([(day, w) for w in wards], columns=["date", "ward_id"])
            rain = replay.ReplayRain(replay.issue_time(day, lead))
            table = flood.build(grid, tide_model=tide_model, rain=rain)
            if len(table) < len(wards):
                continue
            p = model.predict_proba(table[meta["features"]])[:, 1]
            rule = replay.heavy(table)
            for ward, prob, by_rule in zip(table["ward_id"], p, rule, strict=True):
                level = risk.severity(float(prob))
                alerted = at_least(level, RULES[Hazard.FLOOD].alert_from)
                rows.append((day, ward, float(prob), alerted, bool(by_rule)))
    return pd.DataFrame(rows, columns=["date", "ward_id", "probability", "alert", "rule"])


def false_alarms(days: pd.DataFrame, reported: set[tuple[date, str]]) -> pd.DataFrame:
    """Per ward: alert-days, how many a flood report confirms, the unconfirmed share; the same
    for the heavy-rain rule."""
    days = days.copy()
    days["confirmed"] = [
        (d, w) in reported for d, w in zip(days["date"], days["ward_id"], strict=True)
    ]
    out = []
    for ward, g in days.groupby("ward_id"):
        row = {"ward_id": ward, "days": len(g), "reports": int(g["confirmed"].sum())}
        for name in ("alert", "rule"):
            flagged = g[g[name]]
            n, hit = len(flagged), int(flagged["confirmed"].sum())
            row |= {f"{name}_days": n, f"{name}_confirmed": hit}
            row[f"{name}_false"] = None if n == 0 else 1 - hit / n
        out.append(row)
    return pd.DataFrame(out).sort_values("ward_id").reset_index(drop=True)


def save_targets(table: pd.DataFrame, period: str, now: datetime) -> None:
    """Write measured shares; keep any target already set by hand. Heat has no outcome data:
    its targets get the default and stay unmeasured."""
    init_db()
    with SessionLocal.begin() as session:
        for r in table.itertuples():
            for hazard in (Hazard.FLOOD, Hazard.HEAT):
                t = session.get(AlertTarget, (r.ward_id, hazard.value))
                if t is None:
                    t = AlertTarget(
                        ward_id=r.ward_id,
                        hazard=hazard.value,
                        max_false_alarm=DEFAULT_MAX_FALSE_ALARM,
                    )
                    session.add(t)
                if hazard == Hazard.FLOOD:
                    t.alerts, t.confirmed = int(r.alert_days), int(r.alert_confirmed)
                    t.false_alarm = None if pd.isna(r.alert_false) else float(r.alert_false)
                    t.period, t.measured_at = period, now


def set_target(ward_id: str, hazard: Hazard, share: float) -> None:
    if not 0 <= share <= 1:
        raise ValueError("the target is a share from 0 to 1")
    init_db()
    with SessionLocal.begin() as session:
        t = session.get(AlertTarget, (ward_id, hazard.value))
        if t is None:
            session.add(AlertTarget(ward_id=ward_id, hazard=hazard.value, max_false_alarm=share))
        else:
            t.max_false_alarm = share


# --- Report ---------------------------------------------------------------------------------


def _pct(x, digits: int = 0) -> str:
    return "–" if x is None or pd.isna(x) else f"{x:.{digits}%}"


def report(table: pd.DataFrame, period: str, command: str, example: str) -> str:
    over = table[
        (table["alert_days"] >= MIN_ALERT_DAYS) & (table["alert_false"] > DEFAULT_MAX_FALSE_ALARM)
    ]
    total_days = int(table["alert_days"].sum())
    total_hit = int(table["alert_confirmed"].sum())
    rule_days = int(table["rule_days"].sum())
    rule_hit = int(table["rule_confirmed"].sum())
    reports = int(table["reports"].sum())
    lines = [
        "# Alert engine (Stage 13)",
        "",
        f"Built {datetime.now(UTC):%d %b %Y} with `{command}`. Method in `alerts/engine.py`; "
        "signing in `alerts/keys.py`; CAP in `alerts/cap.py`; operators in "
        "`alerts/operators.py`.",
        "",
        "## Severity rules",
        "",
        "| Level in the run | Flood action code | Heat action code | Sent |",
        "| --- | --- | --- | --- |",
        "| green, yellow | – | – | no alert; an active alert is cancelled |",
        "| orange | FLOOD_AVOID_LOW_ROADS | HEAT_LIMIT_SUN | at once |",
        "| red | FLOOD_MOVE_UP | HEAT_STAY_COOL | after an operator approves |",
        "",
        "Also held for an operator: an update that lowers a red alert, a cancel that ends one, a "
        "new alert with low trust (the forecast models disagree, Stage 11) and a new alert in a "
        "ward over its false-alarm target. The levels come from the risk runs: flood "
        "probability bands 0.25 / 0.5 / 0.75 (Stage 6), heat percentiles 90 / 95 / 99 "
        "(Stage 9). Landslide susceptibility (Stage 10) is a static map and sends no alerts.",
        "",
        "## Messages",
        "",
        "- One alert per ward and hazard episode (`alert_id`, e.g. `flood-F-N-20250819-3fa2c1`). "
        "Its messages count up from `sequence` 1: alert, updates, at most one cancel. A "
        "receiver drops a message whose sequence is not above the last it holds, so an old "
        "message replayed later changes nothing.",
        "- Every message expires: at the end of the risk window it is about (a cancel: six "
        "hours after the alert would have expired).",
        "- Ed25519 signature over the JSON without `signature`, keys sorted, no spaces, UTF-8. "
        "`key_id` names the public key (`GET /v1/alert-keys`). The private key is a file on the "
        "machine that runs the engine (`ALERT_SIGNING_KEY_PATH`); the API never opens it. "
        "In the pilot both run on one laptop: the split is in the code, not yet in hardware.",
        "- Every message is TEST unless the run is live and `ALERTS_LIVE=1` is set.",
        "- CAP 1.2: `GET /v1/alerts/{alert_id}/cap`. Status Test or Actual, msgType Alert, "
        "Update or Cancel with `<references>` to the earlier messages, the ward outline as "
        "polygons, BMC ward geocode, and the signature, key_id and sequence as parameters.",
        "",
        "## Operators",
        "",
        "Approval needs a password (scrypt hash) and a six-digit authenticator code (TOTP, "
        "RFC 6238). A code works once; five failures lock the operator for 15 minutes. "
        "Operators can also cancel any active alert by hand. The command line is the only "
        "approval screen until the Stage 22 dashboard.",
        "",
        "## False-alarm target per ward",
        "",
        f"Target: at most {DEFAULT_MAX_FALSE_ALARM:.0%} of a ward's flood alerts unconfirmed by "
        f"a flood report, judged once a ward has {MIN_ALERT_DAYS} or more alert-days. A ward "
        "over its target has its new orange alerts held for an operator. Set a different "
        "target with `--set-target WARD flood SHARE`.",
        "",
        f"Measured on {period}: every day-ahead run from the Stage 7 forecast archive, one "
        f"alert-day for each ward at orange or red. The {reports} reported flooded ward-days in "
        f"those seasons are the only ground truth, so an unconfirmed day is not proof of a "
        f"false alarm: most days have no report either way. The heavy-rain rule (Stage 5 "
        f"baseline) is shown for comparison.",
        "",
        f"All wards: the model alerted on {total_days} ward-days, {total_hit} confirmed "
        f"({_pct(1 - total_hit / total_days if total_days else None, 1)} unconfirmed); the rule on "
        f"{rule_days}, {rule_hit} confirmed "
        f"({_pct(1 - rule_hit / rule_days if rule_days else None, 1)} unconfirmed). "
        f"{len(over)} of {len(table)} wards are over the target.",
        "",
        "| Ward | Days | Reported floods | Model alert-days | Confirmed | Unconfirmed | "
        "Rule days | Rule unconfirmed | Over target |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in table.itertuples():
        is_over = r.alert_days >= MIN_ALERT_DAYS and (r.alert_false or 0) > DEFAULT_MAX_FALSE_ALARM
        lines.append(
            f"| {r.ward_id} | {r.days} | {r.reports} | {r.alert_days} | {r.alert_confirmed} | "
            f"{_pct(r.alert_false)} | {r.rule_days} | {_pct(r.rule_false)} | "
            f"{'yes' if is_over else ''} |"
        )
    lines += [
        "",
        f"What this means: {len(over)} of {len(table)} wards are over the target. Two things "
        "add up. The Stage 5 flood model reaches orange on most monsoon days (its labels have "
        f"only dry-day negatives), {total_days / max(rule_days, 1):.0f} times as often as the "
        "heavy-rain rule. And reports "
        "are so few that even the rule leaves almost every alert-day unconfirmed. Counting "
        "days without a report as false alarms is the cautious reading: over the target, new "
        "flood alerts wait for an operator rather than go out unchecked. Once citizen reports "
        "(Stage 19) give outcomes, including 'no flooding here', the share should be measured "
        "over alert-days with an outcome; the model fix is hard negatives from BMC records.",
        "",
        "## Example message",
        "",
        "A TEST flood alert as receivers get it (signed with a throwaway key for this page):",
        "",
        "```xml",
        example.strip(),
        "```",
        "",
        "## Limits",
        "",
        "- The false-alarm share counts days without a report as unconfirmed. Citizen reports "
        "(Stage 19) and the feedback loop (Stage 20) will add outcomes; until then the share "
        "is an upper bound.",
        "- Heat has no outcome data, so heat targets are set but not measured.",
        "- The English headline and instruction in CAP are placeholders for the reviewed "
        "templates in three languages (Stage 16). Delivery (Telegram, web push, SMS) is "
        "Stage 18.",
        "- The private key file is unencrypted on disk; protect it by file permissions and keep "
        "it off any machine that serves the public API.",
        "",
    ]
    return "\n".join(lines)


def example_cap() -> str:
    private = Ed25519PrivateKey.generate()
    onset = datetime(2025, 8, 18, 18, 30, tzinfo=UTC)
    body = Alert(
        alert_id="flood-F-N-20250819-3fa2c1",
        key_id=keys.key_id(keys.public_raw(private)),
        sequence=1,
        hazard=Hazard.FLOOD,
        ward_id="F-N",
        severity=Severity.ORANGE,
        issued_at=onset - timedelta(hours=10),
        expires_at=onset + timedelta(days=1),
        trust=Trust.MEDIUM,
        action_code="FLOOD_AVOID_LOW_ROADS",
        onset=onset,
        run_id="replay-20250819-lead1d",
    ).model_dump(mode="json")
    body["signature"] = keys.sign(private, body)
    return cap.to_cap(Alert(**body), "F/N", [], [])


# --- Command line ---------------------------------------------------------------------------


def _ask(name: str) -> tuple[str, str]:
    password = getpass.getpass(f"Password for {name}: ")
    code = input("Authenticator code: ")
    return password, code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alert engine.")
    parser.add_argument("--run", help="a run_id, or 'latest' with --hazard")
    parser.add_argument("--hazard", type=Hazard, default=Hazard.FLOOD)
    parser.add_argument("--as-issued", action="store_true", help="act as at the run's issue time")
    parser.add_argument("--pending", action="store_true", help="list drafts waiting for approval")
    parser.add_argument("--approve", type=int, metavar="MESSAGE_ID")
    parser.add_argument("--reject", type=int, metavar="MESSAGE_ID")
    parser.add_argument("--cancel", metavar="ALERT_ID")
    parser.add_argument("--operator")
    parser.add_argument("--reason", default="")
    parser.add_argument("--cap", metavar="ALERT_ID")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--targets", action="store_true", help="measure false-alarm shares")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--set-target", nargs=3, metavar=("WARD", "HAZARD", "SHARE"))
    args = parser.parse_args(argv)

    if args.run:
        with SessionLocal() as session:
            run_id = latest_run(session, args.hazard) if args.run == "latest" else args.run
            if run_id is None:
                print(f"No {args.hazard.value} run yet.")
                return 1
            now = session.get(RiskRun, run_id).issued_at if args.as_issued else None
        out = process_run(run_id, now=now)
        print(
            f"Run {run_id} ({out['hazard']}): {len(out['issued'])} sent, "
            f"{len(out['pending'])} waiting, {out['superseded']} older drafts superseded"
        )
        for alert_id, seq, kind in out["issued"]:
            print(f"  sent     {kind:<6} {alert_id} #{seq}")
        for mid, alert_id, kind, reason in out["pending"]:
            print(f"  waiting  {kind:<6} {alert_id}  message {mid}: {reason}")
    if args.pending:
        with SessionLocal() as session:
            rows = session.scalars(
                select(AlertMessage)
                .where(AlertMessage.status == "pending", AlertMessage.expires_at > func.now())
                .order_by(AlertMessage.id)
            )
            for m in rows:
                print(f"{m.id:>6}  {m.message_type:<6} {m.severity:<6} {m.alert_id}  {m.reason}")
    if args.approve or args.reject or args.cancel:
        if not args.operator:
            parser.error("--approve, --reject and --cancel need --operator")
        password, code = _ask(args.operator)
        try:
            if args.approve:
                alert = approve(args.approve, args.operator, password, code)
                print(f"Sent {alert.message_type.value} {alert.alert_id} #{alert.sequence}")
            elif args.reject:
                reject(args.reject, args.operator, password, code, args.reason or "no reason")
                print(f"Rejected message {args.reject}")
            else:
                alert = cancel(
                    args.cancel, args.operator, password, code, args.reason or "no reason"
                )
                print(f"Sent cancel {alert.alert_id} #{alert.sequence}")
        except (operators.LoginError, ValueError) as exc:
            print(f"Refused: {exc}")
            return 1
    if args.cap:
        from api.db import engine

        with engine.connect() as conn:
            xml = cap_xml(conn, args.cap, args.sequence)
        if args.out:
            args.out.write_text(xml, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(xml)
    if args.set_target:
        ward, hazard, share = args.set_target
        set_target(ward, Hazard(hazard), float(share))
        print(f"Target for {ward} {hazard}: at most {float(share):.0%} unconfirmed")
    if args.targets:
        from pipelines import labels

        table = labels.ward_days(labels.read())
        reported = set(
            zip(
                table.loc[table["flooded"] == 1, "date"],
                table.loc[table["flooded"] == 1, "ward_id"],
                strict=True,
            )
        )
        days = season_alert_days()
        result = false_alarms(days, reported)
        years = TARGET_YEARS
        period = f"June to September {years[0]} and {years[-1]}, day-ahead runs"
        save_targets(result, period, datetime.now(UTC))
        print(result.to_string(index=False))
        if args.report:
            command = "python -m alerts.engine " + " ".join(
                argv if argv is not None else sys.argv[1:]
            )
            args.report.write_text(report(result, period, command, example_cap()), encoding="utf-8")
            print(f"Wrote {args.report}")
    if not any(
        (
            args.run,
            args.pending,
            args.approve,
            args.reject,
            args.cancel,
            args.cap,
            args.targets,
            args.set_target,
        )
    ):
        parser.error("nothing to do; see --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
