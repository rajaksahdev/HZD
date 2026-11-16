"""Offline tests for the schema, range and gap checks and the fault detector."""

from datetime import date

import numpy as np
import pytest

from pipelines import anomaly, qc
from pipelines.grid import empty_dataset

DAY = date(2025, 7, 20)


def daily_rain(value: float = 5.0):
    ds = empty_dataset([np.datetime64(DAY.isoformat())])
    data = np.full((1, ds.sizes["lat"], ds.sizes["lon"]), value, dtype="float32")
    ds["precipitation"] = (("time", "lat", "lon"), data, {"units": "mm"})
    return ds


def hourly_tide(hours: int = 24):
    import xarray as xr

    times = np.datetime64(DAY.isoformat(), "ns") + np.arange(hours) * np.timedelta64(1, "h")
    values = np.sin(np.arange(hours) / 2).astype("float32")
    return xr.Dataset({"sea_level": (("time",), values, {"units": "m"})}, coords={"time": times})


def test_good_data_passes_unchanged():
    ds, issues = qc.validate("chirps", daily_rain(), DAY)
    assert issues == []
    assert ds.attrs["qc_masked_values"] == 0
    assert float(ds["precipitation"].mean()) == 5.0


@pytest.mark.parametrize(
    "break_it, message",
    [
        (lambda ds: ds.drop_vars("precipitation"), "variable is missing"),
        (lambda ds: ds.assign(precipitation=ds["precipitation"].assign_attrs(units="cm")), "units"),
        (lambda ds: ds.assign(extra=ds["precipitation"]), "not in the schema"),
        (lambda ds: ds.assign_coords(lon=ds["lon"] + 0.05), "pilot grid"),
        (lambda ds: ds.assign_coords(time=ds["time"] + np.timedelta64(2, "D")), "outside"),
        (lambda ds: ds.squeeze("time"), "dimensions"),
    ],
)
def test_schema_failures_are_errors(break_it, message):
    with pytest.raises(qc.SchemaError, match=message):
        qc.validate("chirps", break_it(daily_rain()), DAY)


def test_out_of_range_values_are_masked_and_counted():
    ds = daily_rain()
    ds["precipitation"][0, 0, 0] = -3.0
    ds["precipitation"][0, 1, 1] = 5000.0
    cleaned, issues = qc.validate("chirps", ds, DAY)
    assert [(i.check, i.count) for i in issues] == [("range", 2)]
    assert cleaned.attrs["qc_masked_values"] == 2
    assert int(cleaned["precipitation"].isnull().sum()) == 2
    assert cleaned["precipitation"].attrs["units"] == "mm"
    assert float(ds["precipitation"][0, 1, 1]) == 5000.0  # the input is not changed


def test_missing_time_steps_and_values_are_reported():
    _, issues = qc.validate("tide", hourly_tide(hours=20), DAY)
    assert [(i.check, i.count) for i in issues] == [("gap", 4)]

    empty = hourly_tide()
    empty["sea_level"][:] = np.nan
    _, issues = qc.validate("tide", empty, DAY)
    assert issues[0].detail == "every value is missing"


def test_issues_are_logged(tmp_path, monkeypatch):
    from api.config import Settings

    monkeypatch.setattr(qc, "settings", Settings("unused", "unused", (0, 0, 1, 1), tmp_path))
    qc.log_issues("tide", DAY, [qc.Issue("gap", "warning", "", "4 of 24 time steps", 4)])
    qc.log_issues("tide", DAY, [])
    lines = (tmp_path / "processed" / qc.QC_LOG).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and '"source": "tide"' in lines[0]


def test_inject_changes_only_the_fault_window():
    base = np.arange(100, dtype=float)
    spike = anomaly.inject(base, "spike", 50, 1, 9.0)
    assert spike[50] == 59 and np.array_equal(np.delete(spike, 50), np.delete(base, 50))
    flat = anomaly.inject(base, "flat", 50, 6, 0.0)
    assert (flat[50:56] == 49).all() and flat[56] == 56
    drift = anomaly.inject(base, "drift", 50, 10, 5.0)
    assert drift[50] == 50.5 and drift[59] == 64 and drift[60] == 60


def synthetic_series(days: int = 40, cells: int = 6) -> anomaly.Series:
    rng = np.random.default_rng(1)
    t = np.arange(days * 24)
    daily = 28 + 3 * np.sin(2 * np.pi * t / 24)
    values = daily[:, None] + rng.normal(0, 0.3, (t.size, cells)) + np.linspace(0, 1, cells)
    return anomaly.Series("temperature_2m", t % 24, values, "neighbours")


def test_detector_catches_large_faults_and_keeps_false_alarms_low(monkeypatch):
    monkeypatch.setattr(anomaly, "EVENTS", 15)
    result = anomaly.evaluate(synthetic_series())
    rates = {(r["fault"], r["size"]): r["rate"] for r in result["rows"]}
    assert result["false_alarm_rate"] < 0.05
    assert rates[("spike", "8 sigma")] == 1.0
    assert rates[("flat", "24 h")] == 1.0
    assert rates[("drift", "8 sigma over 48 h")] == 1.0
    assert "Detection rate by fault type" in anomaly.report([result], DAY, DAY)
