"""Shared contracts between modules. Every module exchanges these, keyed by ward_id.

Change a schema only by bumping SCHEMA_VERSION and telling the other coder.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class Hazard(StrEnum):
    RAIN = "rain"
    FLOOD = "flood"
    HEAT = "heat"
    LANDSLIDE = "landslide"


class Severity(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


class Trust(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MessageType(StrEnum):
    ALERT = "alert"
    UPDATE = "update"
    CANCEL = "cancel"


class ReportStatus(StrEnum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION


class WardRisk(_Base):
    """One hazard estimate for one ward over one validity window."""

    ward_id: str
    hazard: Hazard
    valid_from: datetime
    valid_to: datetime
    probability: float = Field(ge=0, le=1)
    severity: Severity
    trust: Trust
    run_id: str
    data_age_minutes: int | None = Field(default=None, ge=0)
    degraded: bool = False  # true when a fallback (stale data, threshold rules) produced this

    @model_validator(mode="after")
    def _window(self):
        if self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        return self


class Alert(_Base):
    """Structured alert. Text is rendered later from approved templates via action_code."""

    alert_id: str
    key_id: str
    sequence: int = Field(ge=1)
    message_type: MessageType = MessageType.ALERT
    hazard: Hazard
    ward_id: str
    severity: Severity
    issued_at: datetime
    expires_at: datetime
    trust: Trust
    action_code: str
    is_test: bool = True  # demo alerts stay TEST unless an operator says otherwise
    signature: str | None = None  # Ed25519, added by the alert engine (Stage 13)

    @model_validator(mode="after")
    def _window(self):
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        return self


class CitizenReport(_Base):
    """Ground report from a citizen. `flooding=False` is the 'no flooding here' button."""

    report_id: str
    ts: datetime
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    ward_id: str | None = None
    flooding: bool
    water_level_cm: int | None = Field(default=None, ge=0, le=500)
    text: str | None = Field(default=None, max_length=1000)
    photo_uri: str | None = None
    validation_score: float | None = Field(default=None, ge=0, le=1)
    status: ReportStatus = ReportStatus.PENDING
