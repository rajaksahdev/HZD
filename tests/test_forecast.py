"""Offline tests for the forecast scores and the bias correction."""

import numpy as np
import pytest

from models import forecast


def test_contingency_and_scores():
    obs = np.array([0.0, 10.0, 70.0, 80.0, np.nan])
    fc = np.array([70.0, 0.0, 65.0, 10.0, 99.0])
    assert forecast.contingency(fc, obs, 64.5) == (1, 1, 1)  # the NaN pair is ignored
    s = forecast.scores(1, 1, 1)
    assert s == {"pod": 0.5, "far": 0.5, "csi": pytest.approx(1 / 3), "bias": 1.0}


def test_scores_are_nan_when_undefined():
    s = forecast.scores(0, 0, 0)
    assert all(np.isnan(v) for v in s.values())


def test_quantile_map_removes_a_known_bias():
    rng = np.random.default_rng(0)
    truth = rng.gamma(0.6, 20.0, size=5000)
    biased = truth * 0.5  # the "model" always gives half the rain
    mapping = forecast.fit_quantile_map(biased[:4000], truth[:4000])
    fixed = forecast.apply_quantile_map(mapping, biased[4000:])
    assert abs(fixed.mean() - truth[4000:].mean()) < 0.1 * truth[4000:].mean()
    assert abs(biased[4000:].mean() - truth[4000:].mean()) > 0.4 * truth[4000:].mean()


def test_quantile_map_keeps_order_and_missing_values():
    mapping = forecast.fit_quantile_map(np.linspace(0, 50, 200), np.linspace(0, 100, 200))
    out = forecast.apply_quantile_map(mapping, np.array([0.0, 10.0, 50.0, 200.0, np.nan]))
    assert out[0] == pytest.approx(0, abs=0.5) and out[1] == pytest.approx(20, abs=1)
    assert out[3] == pytest.approx(400)  # beyond the training range: scaled by the top ratio
    assert np.isnan(out[4])
    assert (np.diff(out[:4]) > 0).all()


def test_evaluate_ranks_a_good_forecast_above_persistence():
    rng = np.random.default_rng(1)

    def period(days):
        obs = rng.gamma(0.5, 30.0, size=(days, 3, 3))
        noisy = obs[:, None] * rng.uniform(0.3, 0.7, size=(days, 3, 3, 3))
        return {"obs": obs, "fc": noisy, "persistence": np.roll(obs, 1, axis=0)}

    table, mapping = forecast.evaluate(period(200), period(200))
    heavy = table[table["threshold_mm"] == 64.5].set_index("method")
    assert heavy.loc["corrected mean", "csi"] > heavy.loc["multi-model mean", "csi"]
    assert heavy.loc["corrected mean", "csi"] > heavy.loc["persistence", "csi"]
    assert len(mapping["forecast"]) == len(forecast.QUANTILES)
