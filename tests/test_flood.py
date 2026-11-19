"""Offline tests for the harmonic tide model and the flood model."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from models import flood, tide


def test_tide_fit_recovers_known_constituents():
    times = np.datetime64("2023-01-01T00:00") + np.arange(24 * 400).astype("timedelta64[h]")
    truth = {
        "mean_m": 0.1,
        "constituents": {
            "M2": {"amplitude_m": 1.2, "phase_deg": 40.0},
            "S2": {"amplitude_m": 0.45, "phase_deg": 100.0},
            "K1": {"amplitude_m": 0.4, "phase_deg": 300.0},
            "O1": {"amplitude_m": 0.2, "phase_deg": 210.0},
        },
    }
    heights = tide.predict(truth, times)
    heights[::7] = np.nan  # gaps are skipped
    fitted = tide.fit(times, heights)
    assert fitted["mean_m"] == pytest.approx(0.1, abs=0.01)
    for name, c in truth["constituents"].items():
        got = fitted["constituents"][name]
        assert got["amplitude_m"] == pytest.approx(c["amplitude_m"], abs=0.01)
        assert got["phase_deg"] == pytest.approx(c["phase_deg"], abs=1.0)
    assert fitted["constituents"]["N2"]["amplitude_m"] < 0.01
    # It predicts years outside the fit, which is the point of the model.
    later = np.datetime64("2005-07-26T00:00") + np.arange(48).astype("timedelta64[h]")
    np.testing.assert_allclose(tide.predict(fitted, later), tide.predict(truth, later), atol=0.02)


def test_nodal_factors_follow_the_18_6_year_cycle():
    hours = tide._hours(np.array(["2006-03-01", "2015-06-01"], dtype="datetime64[m]"))
    f, _ = tide.nodal("K1", hours)
    assert f.max() > 1.1 and f.min() < 0.9  # K1 swings about +/-11 % over the cycle
    f_s2, u_s2 = tide.nodal("S2", hours)
    assert (f_s2 == 1).all() and (u_s2 == 0).all()


def test_daily_high_covers_the_ist_day():
    model = {"mean_m": 0.0, "constituents": {"M2": {"amplitude_m": 1.0, "phase_deg": 0.0}}}
    day = date(2024, 7, 8)
    start = np.datetime64("2024-07-07T18:30")  # 00:00 IST
    minutes = start + np.arange(24 * 60).astype("timedelta64[m]")
    expected = tide.predict(model, minutes).max()
    assert tide.daily_high(model, [day])[day] == pytest.approx(expected, abs=0.005)


def test_rain_features_use_the_right_imd_days(monkeypatch):
    day = date(2005, 7, 26)

    def fake(d):  # rain equal to the day offset + 10, the same in every ward
        return pd.Series({"L": float((d - day).days + 10)})

    monkeypatch.setattr(flood, "imd_at_wards", fake)
    out = flood.rain_features(day, "L")
    assert out == {"rain_overnight_mm": 10.0, "rain_today_mm": 11.0, "rain_prior3_mm": 7 + 8 + 9}

    monkeypatch.setattr(flood, "imd_at_wards", lambda d: None if d > day else fake(d))
    assert flood.rain_features(day, "L") is None


def test_scores_and_rule():
    s = flood.scores([1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0.9, 0.2, 0.6, 0.1, 0.0])
    assert (s["hits"], s["misses"], s["false_alarms"], s["correct_no"]) == (1, 1, 1, 2)
    assert s["pod"] == 0.5 and s["far"] == 0.5 and s["pofd"] == pytest.approx(1 / 3)
    assert s["csi"] == pytest.approx(1 / 3)
    assert np.isnan(flood.scores([0, 0], [0, 0])["far"])

    table = pd.DataFrame({"rain_overnight_mm": [70.0, 1.0, 10.0], "rain_today_mm": [0, 64.5, 5]})
    assert flood.rule(table).tolist() == [1.0, 1.0, 0.0]


def synthetic(years) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for year in years:
        for n in range(20):
            wet = n % 2 == 0
            rows.append(
                {
                    "date": date(year, 7, 1) + timedelta(days=n),
                    "ward_id": "L",
                    "year": year,
                    "flooded": int(wet),
                    "rain_overnight_mm": rng.uniform(80, 200) if wet else rng.uniform(0, 2),
                    "rain_today_mm": rng.uniform(50, 150) if wet else rng.uniform(0, 2),
                    "rain_prior3_mm": rng.uniform(0, 100),
                    "tide_high_m": rng.uniform(1, 2.5),
                    **{f: rng.uniform(0, 1) for f in flood.STATIC},
                }
            )
    return pd.DataFrame(rows)


def test_splits_are_by_time(monkeypatch):
    table = synthetic([2005, 2017, 2019, 2024])
    seen = []
    real_fit = flood.fit

    def spy(train, features):
        seen.append(sorted(train["year"].unique()))
        return real_fit(train, features)

    monkeypatch.setattr(flood, "fit", spy)
    by_year = flood.year_by_year(table, flood.FEATURE_SETS["all"])
    assert by_year["year"].tolist() == [2017, 2019, 2024]
    assert seen == [[2005], [2005, 2017], [2005, 2017, 2019]]

    seen.clear()
    results, predictions = flood.held_out(table, 2019)
    assert all(years == [2005, 2017] for years in seen)
    assert set(predictions["date"].map(lambda d: d.year)) == {2019, 2024}
    full = results.set_index("method").loc["XGBoost, all"]
    assert full["pod"] >= 0.9 and full["pofd"] <= 0.1  # noise features can cost a row or two


def test_fit_refuses_one_class():
    table = synthetic([2005])
    with pytest.raises(ValueError):
        flood.fit(table[table["flooded"] == 1], flood.RAIN)


def test_monotone_constraints_line_up_with_features():
    model = flood.make_model(flood.FEATURE_SETS["all"])
    constraints = model.get_params()["monotone_constraints"]
    assert len(constraints) == len(flood.FEATURE_SETS["all"])
    assert constraints[flood.FEATURE_SETS["all"].index("elevation_mean_m")] == -1
    assert constraints[flood.FEATURE_SETS["all"].index("rain_today_mm")] == 1
