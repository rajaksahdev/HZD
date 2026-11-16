"""Settings read from .env. Import `settings` everywhere instead of reading os.environ."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    pilot_bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    data_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def _load() -> Settings:
    bbox = tuple(float(v) for v in os.getenv("PILOT_BBOX", "72.70,18.85,73.10,19.35").split(","))
    if len(bbox) != 4:
        raise ValueError("PILOT_BBOX must be min_lon,min_lat,max_lon,max_lat")
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL", "postgresql+psycopg://hazard:change_me@localhost:5433/hazardlens"
        ),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        pilot_bbox=bbox,
        data_dir=ROOT / "data",
    )


settings = _load()
