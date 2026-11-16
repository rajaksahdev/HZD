"""Create tables and load the 24 BMC administrative wards into PostGIS.

Source: DataMeet Municipal Spatial Data (CC BY-SA 2.5 IN). Safe to re-run.
Usage: python -m scripts.load_wards
"""

import json

import requests
from sqlalchemy import text

from api.config import settings
from api.db import engine, init_db

URL = (
    "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/"
    "master/Mumbai/BMC_Wards.geojson"
)
PATH = settings.raw_dir / "wards" / "BMC_Wards.geojson"
EXPECTED = 24

UPSERT = text(
    """
    INSERT INTO wards (ward_id, name, geometry)
    VALUES (:ward_id, :name, ST_Multi(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON(:geom), 4326))))
    ON CONFLICT (ward_id) DO UPDATE SET name = EXCLUDED.name, geometry = EXCLUDED.geometry
    """
)


def ward_id_from_name(name: str) -> str:
    """'F/N' -> 'F-N' so the id is safe inside URLs like /v1/wards/{id}/risk."""
    return name.strip().upper().replace("/", "-")


def fetch() -> dict:
    if not PATH.exists():
        PATH.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        PATH.write_bytes(resp.content)
    return json.loads(PATH.read_text(encoding="utf-8"))


def main() -> None:
    features = fetch()["features"]
    if len(features) != EXPECTED:
        raise SystemExit(f"Expected {EXPECTED} wards, file has {len(features)}")
    init_db()
    with engine.begin() as conn:
        for f in features:
            name = f["properties"]["name"]
            conn.execute(
                UPSERT,
                {
                    "ward_id": ward_id_from_name(name),
                    "name": name,
                    "geom": json.dumps(f["geometry"]),
                },
            )
        count = conn.execute(text("SELECT count(*) FROM wards")).scalar_one()
    print(f"{count} wards in PostGIS")


if __name__ == "__main__":
    main()
