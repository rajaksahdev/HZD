"""Database tables and engine. Tables are added stage by stage."""

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine, text
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


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(LINK_RUNS)
