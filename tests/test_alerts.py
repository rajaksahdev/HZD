"""Stage 13: alert engine. Offline: signing, TOTP, the rules and CAP. The last test needs the
database: a signed alert is created, approved, updated and cancelled, and read back by API."""

import base64
import secrets
from datetime import UTC, date, datetime, timedelta
from xml.etree import ElementTree as ET

import pandas as pd
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from alerts import cap, engine, keys, operators
from api.schemas import Alert, Hazard, MessageType, Severity, Trust, WardRisk

NOW = datetime(2026, 7, 20, 6, 0, tzinfo=UTC)
START, END = NOW - timedelta(hours=6), NOW + timedelta(hours=18)


def risk(ward: str, level: Severity, trust: Trust = Trust.MEDIUM, **changes) -> WardRisk:
    fields = dict(
        ward_id=ward,
        hazard=Hazard.FLOOD,
        valid_from=START,
        valid_to=END,
        probability=0.9,
        severity=level,
        trust=trust,
        run_id="r1",
    )
    return WardRisk(**(fields | changes))


def signed(private: Ed25519PrivateKey, **changes) -> Alert:
    fields = dict(
        alert_id="flood-A-20260720-abc123",
        key_id=keys.key_id(keys.public_raw(private)),
        sequence=1,
        hazard=Hazard.FLOOD,
        ward_id="A",
        severity=Severity.ORANGE,
        issued_at=NOW,
        expires_at=END,
        trust=Trust.MEDIUM,
        action_code="FLOOD_AVOID_LOW_ROADS",
        onset=START,
        run_id="r1",
    )
    body = Alert(**(fields | changes)).model_dump(mode="json")
    body["signature"] = keys.sign(private, body)
    return Alert(**body)


# --- Signing --------------------------------------------------------------------------------


def test_signature_checks_out_and_any_change_breaks_it():
    private = Ed25519PrivateKey.generate()
    public = base64.b64encode(keys.public_raw(private)).decode()
    body = signed(private).model_dump(mode="json")
    assert keys.verify(public, body)
    for field, value in [
        ("severity", "red"),
        ("sequence", 2),
        ("ward_id", "B"),
        ("is_test", False),
    ]:
        assert not keys.verify(public, body | {field: value})
    assert not keys.verify(public, body | {"signature": None})
    other = base64.b64encode(keys.public_raw(Ed25519PrivateKey.generate())).decode()
    assert not keys.verify(other, body)


def test_canonical_form_ignores_key_order_and_the_signature():
    a = {"b": 1, "a": "ü", "signature": "x"}
    assert keys.canonical(a) == keys.canonical({"a": "ü", "b": 1}) == '{"a":"ü","b":1}'.encode()


def test_key_file_is_never_overwritten(tmp_path):
    path = tmp_path / "k.pem"
    first = keys.generate(path)
    assert keys.public_raw(keys.load(path)) == keys.public_raw(first)
    with pytest.raises(FileExistsError):
        keys.generate(path)
    assert keys.key_id(keys.public_raw(first)).startswith("hzd-")


# --- Two-factor login -----------------------------------------------------------------------

RFC_SECRET = base64.b32encode(b"12345678901234567890").decode()


@pytest.mark.parametrize(
    "t, code",
    [(59, "287082"), (1111111109, "081804"), (1234567890, "005924"), (2000000000, "279037")],
)
def test_totp_matches_rfc_6238(t, code):
    assert operators.code_at(RFC_SECRET, t // operators.STEP) == code


def test_totp_allows_one_step_of_drift_and_no_reuse():
    when = datetime.fromtimestamp(1111111109, UTC)
    step = operators.step_of(when)
    assert operators.matching_step(RFC_SECRET, "081804", when) == step
    late = operators.code_at(RFC_SECRET, step - 1)
    assert operators.matching_step(RFC_SECRET, late, when) == step - 1
    assert (
        operators.matching_step(RFC_SECRET, operators.code_at(RFC_SECRET, step - 2), when) is None
    )
    assert operators.matching_step(RFC_SECRET, "081804", when, after=step) is None  # used once


def test_password_hash():
    stored = operators.hash_password("correct horse")
    assert stored.startswith("scrypt$") and "correct" not in stored
    assert operators.check_password("correct horse", stored)
    assert not operators.check_password("correct hors", stored)
    assert not operators.check_password("x", "garbage")


# --- Rules ----------------------------------------------------------------------------------


def test_orange_is_sent_at_once_and_red_waits():
    drafts = engine.decide(
        [risk("A", Severity.ORANGE), risk("B", Severity.RED), risk("C", Severity.YELLOW)],
        {},
        set(),
        NOW,
    )
    by_ward = {d.ward_id: d for d in drafts}
    assert set(by_ward) == {"A", "B"}
    assert by_ward["A"].message_type == MessageType.ALERT and by_ward["A"].reasons == []
    assert by_ward["A"].action_code == "FLOOD_AVOID_LOW_ROADS"
    assert by_ward["B"].reasons == ["red alert"] and by_ward["B"].action_code == "FLOOD_MOVE_UP"
    assert by_ward["A"].expires_at == END and by_ward["A"].onset == START


def test_low_trust_and_false_alarm_target_hold_new_alerts():
    drafts = engine.decide(
        [risk("A", Severity.ORANGE, Trust.LOW), risk("B", Severity.ORANGE)], {}, {"B"}, NOW
    )
    reasons = {d.ward_id: d.reasons for d in drafts}
    assert reasons["A"] == ["low trust: the forecast models disagree"]
    assert reasons["B"] == ["ward over its false-alarm target"]


def test_updates_and_cancels_follow_the_active_alert():
    private = Ed25519PrivateKey.generate()
    orange, red = signed(private), signed(private, severity=Severity.RED, ward_id="B")
    active = {"A": orange, "B": red}

    same = engine.decide([risk("A", Severity.ORANGE)], active, {"A"}, NOW)
    assert same == []  # nothing new, nothing sent

    later = engine.decide(
        [risk("A", Severity.ORANGE, valid_to=END + timedelta(days=1))], active, {"A"}, NOW
    )
    assert later[0].message_type == MessageType.UPDATE and later[0].reasons == []
    assert later[0].alert_id == orange.alert_id  # target applies to new alerts only

    up = engine.decide([risk("A", Severity.RED)], active, set(), NOW)
    assert up[0].message_type == MessageType.UPDATE and up[0].reasons == ["red alert"]

    down = engine.decide([risk("B", Severity.ORANGE)], active, set(), NOW)
    assert down[0].message_type == MessageType.UPDATE and down[0].reasons == ["lowers a red alert"]

    ended = engine.decide(
        [risk("A", Severity.YELLOW), risk("B", Severity.GREEN)], active, set(), NOW
    )
    kinds = {d.ward_id: (d.message_type, d.reasons, d.action_code) for d in ended}
    assert kinds["A"] == (MessageType.CANCEL, [], "ALL_CLEAR")
    assert kinds["B"] == (MessageType.CANCEL, ["ends a red alert"], "ALL_CLEAR")
    assert all(d.expires_at > NOW for d in ended)


def test_no_alert_for_a_window_already_over_or_for_landslide():
    over = risk(
        "A", Severity.RED, valid_from=NOW - timedelta(days=2), valid_to=NOW - timedelta(days=1)
    )
    slide = risk("B", Severity.RED, hazard=Hazard.LANDSLIDE)
    assert engine.decide([over, slide], {}, set(), NOW) == []


# --- CAP ------------------------------------------------------------------------------------

CAP_ORDER = [
    "identifier", "sender", "sent", "status", "msgType", "source", "scope", "restriction",
    "addresses", "code", "note", "references", "incidents", "info",
]  # fmt: skip
INFO_ORDER = [
    "language", "category", "event", "responseType", "urgency", "severity", "certainty",
    "audience", "eventCode", "effective", "onset", "expires", "senderName", "headline",
    "description", "instruction", "web", "contact", "parameter", "resource", "area",
]  # fmt: skip


def children(el: ET.Element) -> list[str]:
    return [c.tag.split("}")[1] for c in el]


def in_order(tags: list[str], order: list[str]) -> bool:
    ranks = [order.index(t) for t in tags]
    return ranks == sorted(ranks)


def test_cap_alert_and_cancel():
    private = Ed25519PrivateKey.generate()
    first = signed(private)
    ring = [(72.83, 18.93), (72.84, 18.93), (72.84, 18.94), (72.83, 18.94)]
    root = ET.fromstring(cap.to_cap(first, "A", [ring], []))
    ns = {"c": cap.NS}
    assert root.tag == f"{{{cap.NS}}}alert" and in_order(children(root), CAP_ORDER)
    info = root.find("c:info", ns)
    assert in_order(children(info), INFO_ORDER)
    assert root.findtext("c:status", namespaces=ns) == "Test"
    assert root.findtext("c:msgType", namespaces=ns) == "Alert"
    assert root.findtext("c:sent", namespaces=ns) == "2026-07-20T11:30:00+05:30"
    assert info.findtext("c:severity", namespaces=ns) == "Moderate"
    assert info.findtext("c:headline", namespaces=ns).startswith("TEST. Orange")
    polygon = info.findtext("c:area/c:polygon", namespaces=ns).split()
    assert polygon[0] == polygon[-1] == "18.93000,72.83000" and len(polygon) == 5
    params = {
        p.findtext("c:valueName", namespaces=ns): p.findtext("c:value", namespaces=ns)
        for p in info.findall("c:parameter", ns)
    }
    assert params["HazardLens:signature_ed25519"] == first.signature
    assert root.find("c:references", ns) is None

    end = signed(
        private,
        sequence=2,
        message_type=MessageType.CANCEL,
        issued_at=NOW + timedelta(hours=2),
        action_code="ALL_CLEAR",
    )
    root = ET.fromstring(cap.to_cap(end, "A", [], [first, end]))
    assert root.findtext("c:msgType", namespaces=ns) == "Cancel"
    refs = root.findtext("c:references", namespaces=ns)
    assert refs == f"{cap.SENDER},hazardlens-{first.alert_id}-1,2026-07-20T11:30:00+05:30"
    assert root.findtext("c:info/c:responseType", namespaces=ns) == "AllClear"
    with pytest.raises(ValueError):
        cap.to_cap(Alert(**(first.model_dump() | {"signature": None})), "A")


# --- False-alarm share ----------------------------------------------------------------------


def test_false_alarm_share_counts_unconfirmed_alert_days():
    d = [date(2025, 7, n) for n in range(1, 5)]
    days = pd.DataFrame(
        {
            "date": d * 2,
            "ward_id": ["A"] * 4 + ["B"] * 4,
            "probability": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
            "alert": [True, True, True, False, False, False, False, False],
            "rule": [True, False, False, False, False, False, False, False],
        }
    )
    out = engine.false_alarms(days, {(d[0], "A"), (d[3], "B")}).set_index("ward_id")
    assert out.loc["A", "alert_days"] == 3 and out.loc["A", "alert_confirmed"] == 1
    assert out.loc["A", "alert_false"] == pytest.approx(2 / 3)
    assert out.loc["A", "rule_false"] == 0
    assert pd.isna(out.loc["B", "alert_false"]) and out.loc["B", "reports"] == 1


# --- With the database ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def db():
    from api.db import SessionLocal, init_db
    from api.db import engine as db_engine

    try:
        init_db()
        with db_engine.connect() as conn:
            wards = (
                conn.execute(text("SELECT ward_id FROM wards ORDER BY ward_id LIMIT 2"))
                .scalars()
                .all()
            )
    except OperationalError:
        pytest.skip("database not running")
    if len(wards) < 2:
        pytest.skip("wards not loaded")
    tag = secrets.token_hex(4)
    made = {"runs": [], "operator": f"test-op-{tag}", "tag": tag, "wards": wards}
    yield SessionLocal, db_engine, made
    with db_engine.begin() as conn:
        conn.execute(text("DELETE FROM alert_messages WHERE run_id = ANY(:r)"), {"r": made["runs"]})
        conn.execute(text("DELETE FROM risk_runs WHERE run_id = ANY(:r)"), {"r": made["runs"]})
        conn.execute(text("DELETE FROM operators WHERE name = :n"), {"n": made["operator"]})
        if "key_id" in made:
            conn.execute(text("DELETE FROM alert_keys WHERE key_id = :k"), {"k": made["key_id"]})


def save_run(db, run_id: str, levels: dict[str, Severity], issued: datetime, start: datetime):
    from api.db import RiskRun, WardRiskRow

    session_factory, _, made = db
    made["runs"].append(run_id)
    with session_factory.begin() as session:
        session.add(
            RiskRun(
                run_id=run_id,
                issued_at=issued,
                data_time=issued,
                rain_source="test",
                model="test",
                is_test=True,
            )
        )
        session.flush()
        for ward, level in levels.items():
            session.add(
                WardRiskRow(
                    ward_id=ward,
                    hazard="flood",
                    valid_from=start,
                    valid_to=start + timedelta(days=1),
                    probability=0.9,
                    severity=level.value,
                    trust="medium",
                    run_id=run_id,
                )
            )


def test_signed_alert_created_approved_and_cancelled(db, monkeypatch):
    from api import main

    monkeypatch.setattr(engine, "over_target", lambda session, hazard: set())  # measured data

    _, db_engine, made = db
    a, b = made["wards"]
    private = Ed25519PrivateKey.generate()
    made["key_id"] = keys.key_id(keys.public_raw(private))
    # far in the future, so no other alert is active then
    t0 = datetime(2031, 7, 1, 0, 0, tzinfo=UTC)
    day = t0 - timedelta(hours=5, minutes=30)  # 00:00 IST

    run1 = f"test-alerts-{made['tag']}-1"
    save_run(db, run1, {a: Severity.ORANGE, b: Severity.RED}, t0, day)
    out = engine.process_run(run1, now=t0, private=private)
    assert [(kind, seq) for _, seq, kind in out["issued"]] == [("alert", 1)]
    assert len(out["pending"]) == 1 and out["pending"][0][3] == "red alert"
    alert_a = out["issued"][0][0]
    first_draft, red_id = out["pending"][0][:2]

    # A newer run before anyone approves: nothing new for a; the red draft is replaced but
    # keeps its alert_id.
    run1b = f"test-alerts-{made['tag']}-1b"
    save_run(db, run1b, {a: Severity.ORANGE, b: Severity.RED}, t0, day)
    out = engine.process_run(run1b, now=t0 + timedelta(seconds=30), private=private)
    assert out["issued"] == [] and out["superseded"] == 1
    red_message, same_id = out["pending"][0][:2]
    assert same_id == red_id and red_message != first_draft

    client = TestClient(main.app)
    public = {k["key_id"]: k["public_key"] for k in client.get("/v1/alert-keys").json()}
    active = client.get("/v1/alerts", params={"at": t0.isoformat()}).json()
    assert [x["ward_id"] for x in active["alerts"]] == [a]  # the red one is not sent yet
    body = active["alerts"][0]
    assert body["is_test"] and keys.verify(public[body["key_id"]], body)
    assert any(w.startswith("TEST") for w in active["warnings"])

    # Approval needs both factors; a wrong code counts as a failure.
    with db[0].begin() as session:
        secret = operators.add(session, made["operator"], "long enough password", now=t0)
    code = operators.code_at(secret, operators.step_of(t0))
    with pytest.raises(operators.LoginError):
        engine.approve(red_message, made["operator"], "long enough password", "000000", t0, private)
    with pytest.raises(operators.LoginError):
        engine.approve(red_message, made["operator"], "wrong password!", code, t0, private)
    red = engine.approve(red_message, made["operator"], "long enough password", code, t0, private)
    assert red.severity == Severity.RED and red.sequence == 1 and red.ward_id == b
    with pytest.raises(operators.LoginError):  # the same code cannot be used twice
        engine.cancel(
            red.alert_id, made["operator"], "long enough password", code, "x", t0, private
        )

    # Next run: ward a falls to yellow (cancel at once), ward b stays red (nothing new).
    t1 = t0 + timedelta(hours=3)
    run2 = f"test-alerts-{made['tag']}-2"
    save_run(db, run2, {a: Severity.YELLOW, b: Severity.RED}, t1, day)
    out = engine.process_run(run2, now=t1, private=private)
    assert out["issued"] == [(alert_a, 2, "cancel")] and out["pending"] == []

    # An operator cancels the red alert by hand.
    t2 = t1 + timedelta(minutes=5)
    code = operators.code_at(secret, operators.step_of(t2))
    end = engine.cancel(
        red.alert_id, made["operator"], "long enough password", code, "false alarm", t2, private
    )
    assert end.message_type == MessageType.CANCEL and end.sequence == 2
    with pytest.raises(ValueError):  # nothing left to cancel
        t3 = t2 + timedelta(minutes=1)
        code = operators.code_at(secret, operators.step_of(t3))
        engine.cancel(
            red.alert_id, made["operator"], "long enough password", code, "again", t3, private
        )

    assert client.get("/v1/alerts", params={"at": t2.isoformat()}).json()["alerts"] == []
    history = client.get(f"/v1/alerts/{red.alert_id}").json()["messages"]
    assert [(m["sequence"], m["message_type"]) for m in history] == [(1, "alert"), (2, "cancel")]
    assert all(keys.verify(public[m["key_id"]], m) for m in history)
    with db_engine.connect() as conn:
        by = (
            conn.execute(
                text(
                    "SELECT approved_by FROM alert_messages "
                    "WHERE alert_id = :a AND status = 'issued'"
                ),
                {"a": red.alert_id},
            )
            .scalars()
            .all()
        )
    assert by == [made["operator"]] * 2

    xml = client.get(f"/v1/alerts/{red.alert_id}/cap")
    assert xml.status_code == 200 and "application/xml" in xml.headers["content-type"]
    root = ET.fromstring(xml.text)
    ns = {"c": cap.NS}
    assert root.findtext("c:msgType", namespaces=ns) == "Cancel"
    assert f"hazardlens-{red.alert_id}-1" in root.findtext("c:references", namespaces=ns)
    assert root.find("c:info/c:area/c:polygon", ns) is not None
    assert client.get("/v1/alerts/no-such-alert/cap").status_code == 404
