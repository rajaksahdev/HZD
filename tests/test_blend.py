"""Stage 11: blending, trust from model disagreement, bust detection."""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from api import main
from api.schemas import Hazard, Severity, Trust, WardRisk
from models import blend, flood

IDENTITY = {"forecast": [0.0, 1000.0], "observed": [0.0, 1000.0]}
LIMITS = [0.2, 0.5]


def params(weights: dict, events: dict | None = None) -> dict:
    return {
        "maps": {m: IDENTITY for m in weights},
        "weights": weights,
        "events": events or {},
        "trust_limits": LIMITS,
    }


def test_merge_models_prefers_the_newer_ecmwf_grid_and_keeps_the_older_elsewhere():
    new = np.array([np.nan, 5.0])
    old = np.array([1.0, 9.0])
    out = blend.merge_models({"ecmwf_ifs025": new, "ecmwf_ifs04": old, "gfs_seamless": old})
    assert out[:, blend.MODELS.index("ecmwf")].tolist() == [1.0, 5.0]
    assert out[:, blend.MODELS.index("gfs_seamless")].tolist() == [1.0, 9.0]
    assert np.isnan(out[:, blend.MODELS.index("jma_seamless")]).all()


def test_window_is_the_imd_day_ending_03_utc():
    times = np.arange(
        np.datetime64("2025-07-01T00", "h"),
        np.datetime64("2025-07-03T00", "h"),
        np.timedelta64(1, "h"),
    )
    idx = blend.window_index(times, date(2025, 7, 2))
    assert times[idx[0]] == np.datetime64("2025-07-01T04", "h")
    assert times[idx[-1]] == np.datetime64("2025-07-02T03", "h")
    assert len(idx) == 24
    assert blend.window_index(times, date(2025, 7, 1)) is None  # 30 June not in the file


def test_disagreement_needs_two_models_and_grows_with_spread():
    nan = np.nan
    fc = np.array([[10.0, nan, nan], [10.0, 10.0, nan], [10.0, 12.0, nan], [1.0, 80.0, 30.0]])
    spread = blend.disagreement(fc)
    assert np.isnan(spread[0])
    assert spread[1] == 0
    assert 0 < spread[2] < spread[3]


def test_trust_codes_dry_is_high_and_a_single_model_is_low():
    spread = np.array([0.1, 0.3, 0.9, 0.9, np.nan])
    wet = np.array([True, True, True, False, True])
    n = np.array([3, 3, 3, 3, 1])
    assert blend.trust_codes(spread, wet, n, LIMITS).tolist() == [0, 1, 2, 0, 2]


def test_a_bust_is_two_or_more_rain_classes_apart():
    forecast_mm = np.array([5.0, 20.0, 1.0, 70.0, 0.0])
    observed_mm = np.array([80.0, 70.0, 20.0, 130.0, 2.0])
    # rain vs heavy (2 apart), moderate vs heavy (1), dry vs moderate (2), heavy vs very heavy (1)
    assert blend.busts(forecast_mm, observed_mm).tolist() == [True, False, True, False, False]


def test_blend_weights_only_the_models_present():
    fc = np.full((2, len(blend.MODELS)), np.nan)
    gfs, jma = blend.MODELS.index("gfs_seamless"), blend.MODELS.index("jma_seamless")
    fc[0, [gfs, jma]] = [10.0, 40.0]
    fc[1, gfs] = 10.0
    out = blend.predict(params({"gfs_seamless": 3.0, "jma_seamless": 1.0}), fc)
    assert out["blend"][0] == pytest.approx((3 * 10 + 1 * 40) / 4)
    assert out["blend"][1] == pytest.approx(10.0)
    assert out["n"].tolist() == [2, 1]
    assert out["trust"][1] == 2  # one model: low


def test_fit_weights_the_better_model_and_probability_rises_with_rain():
    rng = np.random.default_rng(1)
    obs = rng.gamma(0.6, 25.0, size=(400, 3, 3))
    fc = np.full((*obs.shape, len(blend.MODELS)), np.nan)
    good, poor = blend.MODELS.index("ecmwf"), blend.MODELS.index("gfs_seamless")
    fc[..., good] = obs * rng.uniform(0.8, 1.2, obs.shape)
    fc[..., poor] = obs * rng.uniform(0.1, 3.0, obs.shape)
    fitted = blend.fit(fc, obs)
    assert fitted["weights"]["ecmwf"] > fitted["weights"]["gfs_seamless"]
    assert set(fitted["maps"]) == {"ecmwf", "gfs_seamless"}
    assert fitted["trust_limits"][0] < fitted["trust_limits"][1]
    probe = np.full((2, len(blend.MODELS)), np.nan)
    probe[:, good] = probe[:, poor] = [5.0, 150.0]
    heavy = blend.predict(fitted, probe)["prob"][64.5]
    assert heavy[0] < 0.2 < 0.8 < heavy[1]


def test_reliability_bins_count_every_case():
    prob = np.array([0.05, 0.05, 0.95, 0.95, np.nan])
    event = np.array([0, 1, 1, 1, 0])
    table = blend.reliability(prob, event)
    assert table["cases"].sum() == 4
    assert table.set_index("bin").loc["0.9-1.0", "observed"] == 1.0


def test_ward_trust_is_the_lowest_over_windows(monkeypatch):
    lons, lats = blend.target_coords()
    wards = pd.DataFrame({"lat": [lats[0], lats[-1]], "lon": [lons[0], lons[-1]]}, index=["A", "B"])
    monkeypatch.setattr(flood, "ward_points", lambda: wards)
    monkeypatch.setattr(
        blend, "load_model", lambda: {"rain": params({"gfs_seamless": 1.0, "jma_seamless": 1.0})}
    )
    gfs, jma = blend.MODELS.index("gfs_seamless"), blend.MODELS.index("jma_seamless")
    agree = np.full((lats.size, lons.size, len(blend.MODELS)), np.nan)
    agree[..., gfs] = agree[..., jma] = 20.0
    split = agree.copy()
    split[0, 0, jma] = 200.0  # the models disagree at ward A's cell only
    assert blend.ward_trust([]).tolist() == [Trust.HIGH, Trust.HIGH]
    assert blend.ward_trust([agree]).tolist() == [Trust.HIGH, Trust.HIGH]
    assert blend.ward_trust([agree, split]).tolist() == [Trust.LOW, Trust.HIGH]
    monkeypatch.setattr(blend, "load_model", lambda: None)
    assert blend.ward_trust([agree]) is None  # no Stage 11 model: the caller falls back


def test_heat_spread_check_uses_the_corrected_models():
    from models import heat

    maps = {m: {"forecast": [0.0, 100.0], "era5": [0.0, 100.0]} for m in heat.MODELS}
    values = pd.DataFrame([[36.0, 36.2, 36.4], [34.0, 36.0, 38.0]], columns=heat.MODELS)
    spread = blend.heat_spread(values, maps)
    assert spread == pytest.approx([0.4, 4.0])
    assert blend.heat_trust(spread, [0.8, 1.6]).tolist() == [0, 2]


NOW = datetime(2026, 7, 1, 6, tzinfo=UTC)


def _risk(ward: str, level: Severity, trust: Trust) -> WardRisk:
    return WardRisk(
        ward_id=ward,
        hazard=Hazard.FLOOD,
        valid_from=NOW,
        valid_to=NOW.replace(hour=23),
        probability=0.1,
        severity=level,
        trust=trust,
        run_id="r",
        data_age_minutes=10,
    )


def test_warnings_name_low_trust_wards_and_show_the_flood_caveat_only_with_a_level():
    run = main.RunInfo(
        run_id="r",
        issued_at=NOW,
        data_time=NOW,
        data_age_minutes=10,
        rain_source="forecast",
        model="flood_xgb",
        is_test=False,
        note=None,
        valid_from=NOW,
        valid_to=NOW.replace(hour=23),
    )
    quiet = main.warnings_for(run, [_risk("A", Severity.GREEN, Trust.HIGH)], NOW)
    assert quiet == []
    out = main.warnings_for(
        run, [_risk("A", Severity.ORANGE, Trust.LOW), _risk("B", Severity.GREEN, Trust.HIGH)], NOW
    )
    assert any(w.startswith("Provisional") for w in out)
    assert out[-1] == (
        "Low trust in A: the rain forecast models disagree. Treat levels there as indicative."
    )
