"""Offline tests for replay mode: only runs that existed at the issue time may be used."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from models import flood, replay


def test_window_hours_cover_the_imd_day():
    hours = replay.window_hours(date(2025, 8, 19))
    assert len(hours) == 24
    assert hours[0] == datetime(2025, 8, 18, 4, tzinfo=UTC)
    assert hours[-1] == datetime(2025, 8, 19, 3, tzinfo=UTC)


def test_lags_only_use_runs_made_before_the_issue_time():
    issued = datetime(2025, 8, 18, 18, 30, tzinfo=UTC)  # 00:00 IST on 19 Aug
    hours = [
        issued - timedelta(hours=5),  # already past: newest run (lag 0)
        issued + timedelta(hours=1),  # 1 h ahead: needs a run made at least 1 h before -> lag 1
        issued + timedelta(hours=24),  # exactly a day ahead: lag 1 is just old enough
        issued + timedelta(hours=25),  # lag 2
    ]
    assert replay.lags(hours, issued) == [0, 1, 1, 2]
    for h, k in zip(hours, replay.lags(hours, issued), strict=True):
        # previous_dayK for hour h is made at the latest 24K hours before h
        assert h - timedelta(days=k) <= issued


def test_issue_times_step_back_whole_days_from_midnight_ist():
    assert replay.issue_time(date(2025, 8, 19), 0) == datetime(2025, 8, 18, 18, 30, tzinfo=UTC)
    assert replay.issue_time(date(2025, 8, 19), 3) == datetime(2025, 8, 15, 18, 30, tzinfo=UTC)


def test_lead_time_needs_an_unbroken_run_of_flags():
    assert replay.lead_time({0: True, 1: True, 2: False, 3: True}) == 1
    assert replay.lead_time({0: True, 1: True, 2: True, 3: True}) == 3
    assert replay.lead_time({0: False, 1: True, 2: True, 3: True}) is None


def test_replay_rain_uses_imd_only_once_published(monkeypatch):
    issued = replay.issue_time(date(2025, 8, 19), 1)  # 00:00 IST on 18 Aug
    monkeypatch.setattr(flood, "imd_at_wards", lambda d: pd.Series({"A": 1.0}))
    monkeypatch.setattr(replay, "forecast_window", lambda w, t: np.full((3, 6, 5), 2.0))
    monkeypatch.setattr(
        replay.risk, "_mapping", lambda: {"forecast": [0, 100], "observed": [0, 100]}
    )
    monkeypatch.setattr(flood, "at_wards", lambda field: pd.Series({"A": float(field.mean())}))
    rain = replay.ReplayRain(issued)
    # The IMD day ending 08:30 IST on 17 Aug is published at 12:00 IST on 17 Aug: known.
    assert rain(date(2025, 8, 17))["A"] == 1.0
    # The IMD day ending on 18 Aug is published at 12:00 IST on 18 Aug, after the issue time.
    assert rain(date(2025, 8, 18))["A"] == pytest.approx(2.0)
    assert rain.used == {date(2025, 8, 17): "IMD", date(2025, 8, 18): "forecast"}
    assert rain.data_time() == issued


def test_forecast_window_refuses_hours_beyond_the_archive(monkeypatch):
    times = np.arange("2025-08-01T00", "2025-09-01T00", dtype="datetime64[h]")
    cube = np.ones((replay.MAX_LAG + 1, 3, times.size, 6, 5), dtype="float32")
    monkeypatch.setattr(replay, "load_month", lambda y, m: (times, cube) if m == 8 else None)
    issued = datetime(2025, 8, 18, 18, 30, tzinfo=UTC)
    total = replay.forecast_window(date(2025, 8, 19), issued)
    assert total.shape == (3, 6, 5) and np.allclose(total, 24.0)
    # Issued 6 days earlier, the last hours would need previous_day7, which is not kept.
    assert replay.forecast_window(date(2025, 8, 19), issued - timedelta(days=6)) is None
