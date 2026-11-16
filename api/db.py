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
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    data_age_minutes: Mapped[int | None] = mapped_column(Integer)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
