"""Needs the database running (docker compose up -d db) and wards loaded."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from api.db import engine


@pytest.fixture(scope="module")
def conn():
    try:
        with engine.connect() as c:
            yield c
    except OperationalError:
        pytest.skip("database not running")


def test_24_wards_loaded(conn):
    assert conn.execute(text("SELECT count(*) FROM wards")).scalar_one() == 24


def test_wards_valid_and_inside_pilot_box(conn):
    bad = conn.execute(
        text(
            "SELECT count(*) FROM wards WHERE NOT ST_IsValid(geometry) "
            "OR NOT ST_Within(geometry, ST_MakeEnvelope(72.70, 18.85, 73.10, 19.35, 4326))"
        )
    ).scalar_one()
    assert bad == 0


def test_point_lookup_finds_ward(conn):
    # Dadar area sits in ward G/N
    row = conn.execute(
        text(
            "SELECT ward_id FROM wards "
            "WHERE ST_Contains(geometry, ST_SetSRID(ST_Point(72.8426, 19.0178), 4326))"
        )
    ).scalar_one_or_none()
    assert row == "G-N"
