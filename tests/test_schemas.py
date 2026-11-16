from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from api.schemas import Alert, CitizenReport, WardRisk
from scripts.load_wards import ward_id_from_name

NOW = datetime(2026, 7, 1, 6, 0, tzinfo=UTC)


def risk(**over):
    base = dict(
        ward_id="F-N",
        hazard="flood",
        valid_from=NOW,
        valid_to=NOW + timedelta(hours=3),
        probability=0.7,
        severity="orange",
        trust="medium",
        run_id="run-1",
    )
    return WardRisk(**(base | over))


def test_ward_risk_round_trip():
    r = risk()
    assert WardRisk.model_validate_json(r.model_dump_json()) == r


@pytest.mark.parametrize(
    "bad",
    [{"probability": 1.2}, {"severity": "purple"}, {"valid_to": NOW}, {"unknown_field": 1}],
)
def test_ward_risk_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        risk(**bad)


def test_alert_defaults_to_test_and_checks_expiry():
    fields = dict(
        alert_id="a1",
        key_id="k1",
        sequence=1,
        hazard="flood",
        ward_id="A",
        severity="red",
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=2),
        trust="high",
        action_code="MOVE_TO_HIGHER_FLOOR",
    )
    assert Alert(**fields).is_test is True
    with pytest.raises(ValidationError):
        Alert(**(fields | {"expires_at": NOW}))


def test_no_flooding_report_is_valid():
    r = CitizenReport(report_id="r1", ts=NOW, lat=19.07, lon=72.87, flooding=False)
    assert r.status == "pending"


def test_ward_id_is_url_safe():
    assert ward_id_from_name("F/N") == "F-N"
    assert ward_id_from_name("a") == "A"
