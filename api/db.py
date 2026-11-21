"""Database tables and engine. Tables are added stage by stage."""

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from api.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Ward(Base):
    __tablename__ = "wards"

    ward_id: Mapped[str] = mapped_column(String(8), primary_key=True)  # e.g. "A", "F-N"
    name: Mapped[str] = mapped_column(String(64))  # official name, e.g. "F/N"
    geometry = mapped_column(Geometry("MULTIPOLYGON", srid=4326), nullable=False)
    population: Mapped[int | None] = mapped_column(Integer)  # filled in Stage 12
    vulnerability_index: Mapped[float | None] = mapped_column(Float)  # filled in Stage 12


class RiskRun(Base):
    """One run of a risk model over all wards (Stage 6). The ward rows share its run_id."""

    __tablename__ = "risk_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    data_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # newest input data
    rain_source: Mapped[str] = mapped_column(String(32))  # "observed" or "forecast"
    model: Mapped[str] = mapped_column(String(64))
    is_test: Mapped[bool] = mapped_column(Boolean, default=True)  # hindcasts and demos: TEST
    note: Mapped[str | None] = mapped_column(String(500))


class WardRiskRow(Base):
    __tablename__ = "ward_risk"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ward_id: Mapped[str] = mapped_column(ForeignKey("wards.ward_id"), index=True)
    hazard: Mapped[str] = mapped_column(String(16), index=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    valid_to: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    probability: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(8))
    trust: Mapped[str] = mapped_column(String(8))
    run_id: Mapped[str] = mapped_column(
        ForeignKey("risk_runs.run_id", ondelete="CASCADE"), index=True
    )
    data_age_minutes: Mapped[int | None] = mapped_column(Integer)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    value: Mapped[float | None] = mapped_column(Float)  # Stage 9: e.g. apparent temperature
    unit: Mapped[str | None] = mapped_column(String(16))


class WardStatic(Base):
    """Static features per ward, written by pipelines/static.py (Stage 3)."""

    __tablename__ = "ward_static"

    ward_id: Mapped[str] = mapped_column(ForeignKey("wards.ward_id"), primary_key=True)
    area_km2: Mapped[float] = mapped_column(Float)
    elevation_mean_m: Mapped[float] = mapped_column(Float)
    elevation_p10_m: Mapped[float] = mapped_column(Float)
    low_lying_frac: Mapped[float] = mapped_column(Float)  # share of the ward below 5 m
    slope_mean_deg: Mapped[float] = mapped_column(Float)
    drain_distance_mean_m: Mapped[float | None] = mapped_column(Float)
    drain_km_per_km2: Mapped[float] = mapped_column(Float)
    built_up_frac: Mapped[float] = mapped_column(Float)
    green_frac: Mapped[float] = mapped_column(Float)
    water_frac: Mapped[float] = mapped_column(Float)
    wetland_frac: Mapped[float] = mapped_column(Float)


class WardVulnerability(Base):
    """Population, access and vulnerability per ward, written by models/impact.py (Stage 12)."""

    __tablename__ = "ward_vulnerability"

    ward_id: Mapped[str] = mapped_column(ForeignKey("wards.ward_id"), primary_key=True)
    population: Mapped[int] = mapped_column(Integer)
    area_km2: Mapped[float] = mapped_column(Float)
    density_per_km2: Mapped[float] = mapped_column(Float)
    low_lying_pop: Mapped[int] = mapped_column(Integer)  # people on land below 5 m
    low_lying_share: Mapped[float] = mapped_column(Float)
    shelter_km: Mapped[float] = mapped_column(Float)  # mean road distance, weighted by people
    hospital_km: Mapped[float] = mapped_column(Float)
    shelter_far_share: Mapped[float] = mapped_column(Float)  # share over 1 km from a shelter
    hospital_far_share: Mapped[float] = mapped_column(Float)  # share over 2 km from a hospital
    vulnerability_index: Mapped[float] = mapped_column(Float)  # mean of ranks, 0..1
    built_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Facility(Base):
    """Hospitals and shelter candidates from OpenStreetMap, or an official shelter list."""

    __tablename__ = "facilities"

    facility_id: Mapped[str] = mapped_column(String(40), primary_key=True)  # e.g. osm-way-123
    kind: Mapped[str] = mapped_column(String(24), index=True)  # hospital, school, ...
    name: Mapped[str | None] = mapped_column(String(200))
    ward_id: Mapped[str | None] = mapped_column(ForeignKey("wards.ward_id"), index=True)
    emergency: Mapped[bool] = mapped_column(Boolean, default=False)  # OSM emergency=yes
    official: Mapped[bool] = mapped_column(Boolean, default=False)  # from an official list
    geometry = mapped_column(Geometry("POINT", srid=4326), nullable=False)


class WardRoute(Base):
    """Road route from a ward to its nearest shelter candidate or hospital for one run (or
    run_id 'baseline': no roads avoided). Not linked to risk_runs so the baseline fits."""

    __tablename__ = "ward_route"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ward_id: Mapped[str] = mapped_column(ForeignKey("wards.ward_id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # shelter or hospital
    facility_id: Mapped[str | None] = mapped_column(String(40))
    facility_name: Mapped[str | None] = mapped_column(String(200))
    distance_m: Mapped[float | None] = mapped_column(Float)  # None: no safe route
    baseline_m: Mapped[float | None] = mapped_column(Float)  # same trip avoiding nothing
    status: Mapped[str] = mapped_column(String(16))
    avoided_km: Mapped[float] = mapped_column(Float, default=0.0)  # road avoided in the ward
    people_cut_off: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    geometry = mapped_column(Geometry("LINESTRING", srid=4326), nullable=True)


class AlertKey(Base):
    """Public half of each alert signing key (Stage 13). The private half never enters the
    database or the API process."""

    __tablename__ = "alert_keys"

    key_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    public_key: Mapped[str] = mapped_column(String(64))  # raw 32 bytes, base64
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AlertMessage(Base):
    """One message of one alert (Stage 13): issued and signed, or a draft waiting for an
    operator. `body` is the exact signed JSON that receivers get."""

    __tablename__ = "alert_messages"
    __table_args__ = (
        Index(
            "alert_messages_issued_seq",
            "alert_id",
            "sequence",
            unique=True,
            postgresql_where=text("status = 'issued'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(String(40), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    message_type: Mapped[str] = mapped_column(String(8))
    hazard: Mapped[str] = mapped_column(String(16), index=True)
    ward_id: Mapped[str] = mapped_column(ForeignKey("wards.ward_id"), index=True)
    severity: Mapped[str] = mapped_column(String(8))
    trust: Mapped[str] = mapped_column(String(8))
    action_code: Mapped[str] = mapped_column(String(32))
    onset: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_test: Mapped[bool] = mapped_column(Boolean, default=True)
    run_id: Mapped[str | None] = mapped_column(String(64))
    # issued, pending (needs an operator), superseded, withdrawn, rejected, expired
    status: Mapped[str] = mapped_column(String(12), index=True)
    reason: Mapped[str | None] = mapped_column(String(300))  # why it needs or got approval
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    approved_by: Mapped[str | None] = mapped_column(String(64))
    key_id: Mapped[str | None] = mapped_column(String(32))
    signature: Mapped[str | None] = mapped_column(String(128))
    body: Mapped[str | None] = mapped_column(Text)


class Operator(Base):
    """A person who may approve red alerts (Stage 13): password and a TOTP authenticator."""

    __tablename__ = "operators"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    totp_secret: Mapped[str] = mapped_column(String(64))
    last_step: Mapped[int] = mapped_column(Integer, default=0)  # a code works only once
    failures: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AlertTarget(Base):
    """False-alarm target per ward and hazard (Stage 13), and the share of past alerts that no
    flood report confirmed. Over the target, even orange alerts need an operator."""

    __tablename__ = "alert_targets"

    ward_id: Mapped[str] = mapped_column(ForeignKey("wards.ward_id"), primary_key=True)
    hazard: Mapped[str] = mapped_column(String(16), primary_key=True)
    max_false_alarm: Mapped[float] = mapped_column(Float)  # target share of unconfirmed alerts
    alerts: Mapped[int | None] = mapped_column(Integer)  # alert-days in the measured period
    confirmed: Mapped[int | None] = mapped_column(Integer)  # of them, with a flood report
    false_alarm: Mapped[float | None] = mapped_column(Float)  # 1 - confirmed / alerts
    period: Mapped[str | None] = mapped_column(String(64))
    measured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ward_risk was created in Stage 0 without the link to risk_runs; create_all never alters a table.
LINK_RUNS = text(
    """
    DO $$ BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ward_risk_run_id_fkey') THEN
        ALTER TABLE ward_risk ADD CONSTRAINT ward_risk_run_id_fkey
          FOREIGN KEY (run_id) REFERENCES risk_runs (run_id) ON DELETE CASCADE;
      END IF;
    END $$
    """
)


# Stage 9 added value and unit to ward_risk; add them to a table created before that.
ADD_VALUE = text(
    "ALTER TABLE ward_risk ADD COLUMN IF NOT EXISTS value double precision, "
    "ADD COLUMN IF NOT EXISTS unit varchar(16)"
)


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(LINK_RUNS)
        conn.execute(ADD_VALUE)
