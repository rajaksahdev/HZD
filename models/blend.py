"""Blending and bust detection (Stage 11): combine the forecast models, give every forecast a
trust level from how much the models disagree, and check that low trust really means more busts.

Archive      Open-Meteo's archive of stitched short-lead forecasts (about a day ahead), June to
             September 2021-2025. Models join as the archive gains them: GFS and JMA from 2021,
             UKMO from 2022, ECMWF, ICON and GEM from 2023. ECMWF is IFS 0.4 degree until
             February 2024 and IFS 0.25 degree after; both count as one model.
Truth        IMD gridded daily rain, the 24 hours ending 08:30 IST (03:00 UTC).
Blend        Each model is quantile-mapped to IMD on its own, then the corrected models present
             that day are averaged with weights 1 / MAE (learned on the training years).
Probability  Chance of moderate (15.6 mm) and heavy (64.5 mm) rain: a logistic regression on the
             share of corrected models above the amount and log(1 + blend).
Trust        Disagreement = spread (standard deviation) of log(1 + corrected rain) across the
             models present. High below the median of the training years' wet cell-days, low
             above their 85th percentile, medium between. A day every model calls dry
             (< 2.5 mm) is high trust; a day with fewer than two models is low.
Bust         The blend and IMD land two or more IMD rain classes apart (dry, rain, moderate,
             heavy, very heavy): for example light rain forecast and heavy rain observed.
Split        Leave one year out: each monsoon 2021-2025 is predicted by a fit on the other four.
             A forward check trains on 2021-2024 and tests on 2025 only. The saved model, used
             by the risk runs, is fitted on all five years.
Heat         The same idea checked for the Stage 9 heat index (spread of the three corrected
             models, limits from the heat model's fitting period, scored on 2025). It does not
             grade heat forecasts, so heat runs do not use it; see the report.

python -m models.blend --fetch 2021 2025       archived forecasts, one request per month
python -m models.blend --report docs/blend.md  fit, score, plot docs/blend_reliability.svg
"""

import argparse
import calendar
import json
import sys
import warnings
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LogisticRegression

from api.config import settings
from api.schemas import Trust
from models import forecast
from pipelines import store
from pipelines.grid import target_coords
from pipelines.sources import openmeteo

NAME = "blend"
URL = openmeteo.ARCHIVE_URL
RAW = settings.raw_dir / NAME
MODEL_PATH = settings.processed_dir / "models" / "blend.json"
API_MODELS = [
    "ecmwf_ifs025",
    "ecmwf_ifs04",
    "gfs_seamless",
    "icon_seamless",
    "jma_seamless",
    "ukmo_seamless",
    "gem_seamless",
]
MODELS = ["ecmwf", "gfs_seamless", "icon_seamless", "jma_seamless", "ukmo_seamless", "gem_seamless"]
ALIAS = {"ecmwf_ifs025": "ecmwf", "ecmwf_ifs04": "ecmwf"}  # newer first: it wins where both exist
LABELS = {
    "ecmwf": "ECMWF",
    "gfs_seamless": "GFS",
    "icon_seamless": "ICON",
    "jma_seamless": "JMA",
    "ukmo_seamless": "UKMO",
    "gem_seamless": "GEM",
}
YEARS = list(range(2021, 2026))
MONTHS = (6, 7, 8, 9)
EVENTS = {15.6: "moderate", 64.5: "heavy"}  # IMD classes, mm in 24 h
CLASSES = ["dry", "rain", "moderate", "heavy", "very heavy"]  # split at forecast.THRESHOLDS
WET = forecast.THRESHOLDS[0]
TRUST_PERCENTILES = (50, 85)
BUST_CLASSES = 2
MIN_CASES = 300  # cell-days a model needs in the training years to join the blend
BINS = np.linspace(0, 1, 11)
MIN_BIN = 30  # cell-days a probability bin needs to be drawn (the tables list every bin)
BOOTSTRAP = 500
LEVELS = [Trust.HIGH, Trust.MEDIUM, Trust.LOW]  # index = trust code
HEAT_TEST = (date(2025, 1, 1), date(2025, 12, 31))


# --- Archive ----------------------------------------------------------------------------------


def _month_file(year: int, month: int) -> Path | None:
    files = sorted((RAW / f"{year}" / f"{month:02d}").glob("blend_*.json"))
    return files[-1] if files else None


def fetch_month(year: int, month: int, force: bool = False) -> Path:
    """Hourly rain from every model for one month, plus the last day of the month before (the
    first IMD day of the month starts at 04:00 UTC the day before)."""
    have = _month_file(year, month)
    if have and not force:
        return have
    lats, lons = openmeteo._points()
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    resp = store.http_get(
        URL,
        params={
            "latitude": ",".join(f"{v:.2f}" for v in lats),
            "longitude": ",".join(f"{v:.2f}" for v in lons),
            "hourly": "precipitation",
            "models": ",".join(API_MODELS),
            "start_date": (first - timedelta(days=1)).isoformat(),
            "end_date": last.isoformat(),
            "timezone": "UTC",
        },
        timeout=300,
        retries=7,  # the archive rate-limits a burst of months; backoff waits up to ~5 minutes
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RAW / f"{year}" / f"{month:02d}" / f"blend_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    store.log_download(NAME, first, path, resp.url)
    return path


def merge_models(series: dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-API-model arrays into MODELS order (last axis), merging the two ECMWF grids."""
    shape = next(iter(series.values())).shape
    out = np.full((*shape, len(MODELS)), np.nan, dtype="float64")
    for name in reversed(API_MODELS):  # older ECMWF first, so the newer one overwrites it
        if name not in series:
            continue
        values = np.asarray(series[name], dtype="float64")
        m = MODELS.index(ALIAS.get(name, name))
        out[..., m] = np.where(np.isfinite(values), values, out[..., m])
    return out


@lru_cache(maxsize=8)
def load_month(year: int, month: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(times, cube) for one month file. cube is (time, lat, lon, model) in mm per hour."""
    path = _month_file(year, month)
    if path is None:
        return None
    points = json.loads(path.read_text(encoding="utf-8"))
    points = [points] if isinstance(points, dict) else points
    lons, lats = target_coords()
    if len(points) != lats.size * lons.size:
        raise ValueError(f"{path}: expected {lats.size * lons.size} points, got {len(points)}")
    times = np.asarray(points[0]["hourly"]["time"], dtype="datetime64[h]")
    series = {}
    for name in API_MODELS:
        grid = np.full((times.size, lats.size, lons.size), np.nan)
        for p, point in enumerate(points):  # lat-major, as openmeteo._points()
            values = point["hourly"].get(f"precipitation_{name}")
            if values is not None:
                grid[:, p // lons.size, p % lons.size] = [
                    np.nan if v is None else v for v in values
                ]
        series[name] = grid
    return times, merge_models(series)


def window_index(times: np.ndarray, day: date) -> np.ndarray | None:
    """Positions of the 24 hourly steps of the IMD day ending 03:00 UTC on `day`."""
    end = np.datetime64(f"{day.isoformat()}T{forecast.OBS_END_HOUR:02d}", "h")
    want = end - np.arange(23, -1, -1).astype("timedelta64[h]")
    idx = np.searchsorted(times, want)
    if (idx >= times.size).any() or (times[idx] != want).any():
        return None
    return idx


def daily_sum(cube: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sum of the hours in `idx`; NaN for a model and cell with any hour missing."""
    return cube[idx].sum(axis=0)  # NaN propagates


def load_season(year: int) -> dict:
    """Matched arrays for June-September: obs (day, lat, lon), fc (day, lat, lon, model)."""
    days, obs, fc = [], [], []
    for month in MONTHS:
        loaded = load_month(year, month)
        if loaded is None:
            raise FileNotFoundError(
                f"No archived forecasts for {year}-{month:02d}. "
                f"Run: python -m models.blend --fetch {year} {year}"
            )
        times, cube = loaded
        for d in range(1, calendar.monthrange(year, month)[1] + 1):
            day = date(year, month, d)
            o, idx = forecast.observed(day), window_index(times, day)
            if o is None or idx is None:
                continue
            days.append(day)
            obs.append(o)
            fc.append(daily_sum(cube, idx))
    if not days:
        raise RuntimeError(f"No IMD days for {year}. Run: python -m pipelines.run ... imd")
    return {"days": days, "obs": np.stack(obs), "fc": np.stack(fc)}


# --- Fit and predict ------------------------------------------------------------------------


def disagreement(corrected: np.ndarray) -> np.ndarray:
    """Spread of log(1 + rain) across the models present (last axis); NaN with fewer than two."""
    logs = np.log1p(np.maximum(corrected, 0))
    n = np.isfinite(logs).sum(axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells: handled below
        spread = np.nanstd(logs, axis=-1)
    return np.where(n >= 2, spread, np.nan)


def trust_codes(spread: np.ndarray, wet: np.ndarray, n: np.ndarray, limits) -> np.ndarray:
    """0 high, 1 medium, 2 low (index into LEVELS)."""
    code = np.where(spread < limits[0], 0, np.where(spread < limits[1], 1, 2))
    code = np.where(wet, code, 0)  # every model calls it dry: they agree
    return np.where(n >= 2, code, 2)


def rain_class(values: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(forecast.THRESHOLDS), values, side="right")


def busts(blend: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """True where forecast and observed rain are two or more IMD classes apart."""
    return np.abs(rain_class(blend) - rain_class(obs)) >= BUST_CLASSES


def correct(params: dict, fc: np.ndarray) -> np.ndarray:
    """Each model quantile-mapped to IMD; NaN for models the fit did not include."""
    out = np.full(fc.shape, np.nan)
    for m, name in enumerate(MODELS):
        if name in params["maps"]:
            out[..., m] = forecast.apply_quantile_map(params["maps"][name], fc[..., m])
    return out


def _features(share: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return np.column_stack([share.ravel(), np.log1p(np.maximum(blend, 0)).ravel()])


def predict(params: dict, fc: np.ndarray) -> dict:
    """Blend, event probabilities, disagreement and trust for forecasts with models last."""
    corrected = correct(params, fc)
    present = np.isfinite(corrected)
    n = present.sum(axis=-1)
    weights = np.array([params["weights"].get(m, 0.0) for m in MODELS])
    w = np.where(present, weights, 0.0)
    total = w.sum(axis=-1)
    blend = np.where(
        total > 0,
        np.nansum(np.where(present, corrected, 0) * w, axis=-1) / np.maximum(total, 1e-12),
        np.nan,
    )
    out = {"corrected": corrected, "blend": blend, "n": n, "prob": {}, "share": {}}
    for amount, event in params["events"].items():
        above = (corrected >= float(amount)) & present
        share = np.where(n > 0, above.sum(axis=-1) / np.maximum(n, 1), np.nan)
        z = _features(np.nan_to_num(share), np.nan_to_num(blend)) @ np.asarray(event["coef"])
        prob = 1 / (1 + np.exp(-(z + event["intercept"])))
        out["share"][float(amount)] = share
        out["prob"][float(amount)] = np.where(n > 0, prob.reshape(blend.shape), np.nan)
    out["spread"] = disagreement(corrected)
    out["wet"] = ((corrected >= WET) & present).any(axis=-1)
    out["trust"] = trust_codes(out["spread"], out["wet"], n, params["trust_limits"])
    return out


def fit(fc: np.ndarray, obs: np.ndarray) -> dict:
    """Quantile maps and weights per model, event regressions and trust limits."""
    params = {"maps": {}, "weights": {}, "events": {}, "trust_limits": [np.inf, np.inf]}
    for m, name in enumerate(MODELS):
        f = fc[..., m]
        ok = np.isfinite(f) & np.isfinite(obs)
        if ok.sum() < MIN_CASES:
            continue
        mapping = forecast.fit_quantile_map(f[ok], obs[ok])
        error = np.abs(forecast.apply_quantile_map(mapping, f[ok]) - obs[ok]).mean()
        params["maps"][name] = mapping
        params["weights"][name] = float(1 / max(error, 0.1))
    if not params["maps"]:
        raise RuntimeError("no model has enough training days")
    base = predict({**params, "events": {}}, fc)
    blend, n = base["blend"], base["n"]
    ok = np.isfinite(blend) & np.isfinite(obs) & (n > 0)
    for amount in EVENTS:
        present = np.isfinite(base["corrected"])
        share = ((base["corrected"] >= amount) & present).sum(axis=-1) / np.maximum(n, 1)
        x = _features(share[ok], blend[ok])
        y = (obs[ok] >= amount).astype(int)
        if y.min() == y.max():
            raise RuntimeError(f"training years have no variety at {amount} mm")
        reg = LogisticRegression().fit(x, y)
        params["events"][str(amount)] = {
            "coef": reg.coef_[0].round(5).tolist(),
            "intercept": round(float(reg.intercept_[0]), 5),
            "climatology": round(float(y.mean()), 5),
        }
    spread = base["spread"][ok & base["wet"]]
    spread = spread[np.isfinite(spread)]
    params["trust_limits"] = [round(float(np.percentile(spread, p)), 4) for p in TRUST_PERCENTILES]
    return params


def stack(seasons: dict[int, dict], years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([seasons[y]["fc"] for y in years]),
        np.concatenate([seasons[y]["obs"] for y in years]),
    )


def plain_mean_map(fc: np.ndarray, obs: np.ndarray) -> dict:
    """Stage 4's way, on the same years: one quantile map for the plain mean of the models."""
    return forecast.fit_quantile_map(_plain_mean(fc), obs)


def _plain_mean(fc: np.ndarray) -> np.ndarray:
    n = np.isfinite(fc).sum(axis=-1)
    return np.where(n > 0, np.nansum(fc, axis=-1) / np.maximum(n, 1), np.nan)


# --- Evaluation -------------------------------------------------------------------------------


def held_out(seasons: dict[int, dict], train_years: dict[int, list[int]]) -> dict:
    """Predictions for each test year from a fit on its training years, stacked by day."""
    parts = []
    for year, others in train_years.items():
        fc, obs = stack(seasons, others)
        params = fit(fc, obs)
        test = seasons[year]
        pred = predict(params, test["fc"])
        stage4 = forecast.apply_quantile_map(plain_mean_map(fc, obs), _plain_mean(test["fc"]))
        clim = {a: params["events"][str(a)]["climatology"] for a in EVENTS}
        parts.append(
            {
                "year": np.full(len(test["days"]), year),
                "days": np.array(test["days"]),
                "obs": test["obs"],
                "raw_mean": _plain_mean(test["fc"]),
                "stage4": stage4,
                "blend": pred["blend"],
                "n": pred["n"],
                "spread": pred["spread"],
                "trust": pred["trust"],
                "corrected": pred["corrected"],
                **{f"prob_{a}": pred["prob"][a] for a in EVENTS},
                **{f"share_{a}": pred["share"][a] for a in EVENTS},
                **{f"clim_{a}": np.full(test["obs"].shape, clim[a]) for a in EVENTS},
                "params": params,
            }
        )
    out = {}
    for key in parts[0]:
        if key == "params":
            out["params"] = {int(p["year"][0]): p["params"] for p in parts}
        else:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


def _day_bootstrap(per_day: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """90% interval of sum(col 0) / sum(col 1) from resampling whole days."""
    if len(per_day) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_day), size=(BOOTSTRAP, len(per_day)))
    totals = per_day[picks].sum(axis=1)
    ratio = totals[:, 0] / np.maximum(totals[:, 1], 1)
    return float(np.quantile(ratio, 0.05)), float(np.quantile(ratio, 0.95))


def method_scores(h: dict) -> pd.DataFrame:
    rows = []
    obs = h["obs"]
    for key, label in (
        ("raw_mean", "plain mean of the models present, uncorrected"),
        ("stage4", "one quantile map on the plain mean (Stage 4's way)"),
        ("blend", "blend: each model mapped, weighted by skill (Stage 11)"),
    ):
        fc = h[key]
        row = {"method": label, "mae": float(np.nanmean(np.abs(fc - obs)))}
        for amount, event in EVENTS.items():
            hits, misses, false_alarms = forecast.contingency(fc, obs, amount)
            sc = forecast.scores(hits, misses, false_alarms)
            low, high = forecast.csi_interval(fc, obs, amount)
            row |= {
                f"{event}_csi": sc["csi"],
                f"{event}_range": (low, high),
                f"{event}_bias": sc["bias"],
            }
        ok = np.isfinite(fc) & np.isfinite(obs)
        row["bust_rate"] = float(busts(fc[ok], obs[ok]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def reliability(prob: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Per probability bin: cases, mean forecast probability, observed frequency."""
    ok = np.isfinite(prob)
    p, e = prob[ok], event[ok]
    which = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    rows = []
    for b in range(len(BINS) - 1):
        inside = which == b
        if inside.any():
            rows.append(
                {
                    "bin": f"{BINS[b]:.1f}-{BINS[b + 1]:.1f}",
                    "cases": int(inside.sum()),
                    "forecast": float(p[inside].mean()),
                    "observed": float(e[inside].mean()),
                }
            )
    return pd.DataFrame(rows)


def brier(prob: np.ndarray, event: np.ndarray, clim: np.ndarray) -> dict:
    ok = np.isfinite(prob) & np.isfinite(clim)
    bs = float(np.mean((prob[ok] - event[ok]) ** 2))
    ref = float(np.mean((clim[ok] - event[ok]) ** 2))
    return {"brier": bs, "skill": 1 - bs / ref if ref > 0 else float("nan")}


def trust_table(h: dict) -> pd.DataFrame:
    """Per trust level: cell-days, MAE, bust rate (90% range over days), heavy-rain misses."""
    ok = np.isfinite(h["blend"]) & np.isfinite(h["obs"])
    bust = busts(np.nan_to_num(h["blend"]), np.nan_to_num(h["obs"])) & ok
    rows = []
    for code, level in enumerate(LEVELS):
        sel = (h["trust"] == code) & ok
        per_day = np.column_stack(
            [(bust & sel).reshape(len(sel), -1).sum(1), sel.reshape(len(sel), -1).sum(1)]
        )
        heavy_obs = (h["obs"] >= 64.5) & sel
        rows.append(
            {
                "trust": level.value,
                "cases": int(sel.sum()),
                "share": float(sel.sum() / ok.sum()),
                "mae": float(np.abs(h["blend"][sel] - h["obs"][sel]).mean())
                if sel.any()
                else np.nan,
                "busts": int((bust & sel).sum()),
                "bust_rate": float((bust & sel).sum() / max(sel.sum(), 1)),
                "bust_range": _day_bootstrap(per_day[per_day[:, 1] > 0]),
                "heavy_seen": int(heavy_obs.sum()),
                "heavy_missed": int((heavy_obs & (h["blend"] < 64.5)).sum()),
            }
        )
    return pd.DataFrame(rows)


def trust_by_amount(h: dict) -> pd.DataFrame:
    """Bust rate by trust within each blended rain class: is trust more than 'it rained a lot'?"""
    ok = np.isfinite(h["blend"]) & np.isfinite(h["obs"])
    bust = busts(np.nan_to_num(h["blend"]), np.nan_to_num(h["obs"]))
    klass = np.minimum(rain_class(np.nan_to_num(h["blend"])), 3)  # heavy and very heavy together
    names = ["dry", "rain", "moderate", "heavy or more"]
    rows = []
    for k, name in enumerate(names):
        row = {"forecast": name}
        for code, level in enumerate(LEVELS):
            sel = ok & (klass == k) & (h["trust"] == code)
            row[level.value] = (int((bust & sel).sum()), int(sel.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def per_year(h: dict, seasons: dict[int, dict]) -> pd.DataFrame:
    rows = []
    ok_all = np.isfinite(h["blend"]) & np.isfinite(h["obs"])
    bust = busts(np.nan_to_num(h["blend"]), np.nan_to_num(h["obs"]))
    for year in sorted(seasons):
        inside = (h["year"] == year)[:, None, None] & ok_all
        fc = seasons[year]["fc"]
        models = [LABELS[m] for i, m in enumerate(MODELS) if np.isfinite(fc[..., i]).mean() > 0.5]
        row = {"year": year, "days": len(seasons[year]["days"]), "models": ", ".join(models)}
        for code, level in enumerate(LEVELS):
            sel = inside & (h["trust"] == code)
            row[level.value] = (int((bust & sel).sum()), int(sel.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def low_trust_days(h: dict, year: int, share: float = 0.5) -> pd.DataFrame:
    """Days of `year` with at least `share` of the cells at low trust, city means."""
    rows = []
    for i in np.flatnonzero(h["year"] == year):
        low = (h["trust"][i] == 2).mean()
        if low < share:
            continue
        corrected = np.nanmean(h["corrected"][i], axis=(0, 1))
        present = [(MODELS[m], v) for m, v in enumerate(corrected) if np.isfinite(v)]
        ok = np.isfinite(h["blend"][i]) & np.isfinite(h["obs"][i])
        rows.append(
            {
                "date": h["days"][i],
                "low_share": float(low),
                "models": ", ".join(f"{LABELS[m]} {v:.0f}" for m, v in present),
                "blend": float(np.nanmean(h["blend"][i])),
                "imd": float(np.nanmean(h["obs"][i])),
                "busts": int(busts(h["blend"][i][ok], h["obs"][i][ok]).sum()),
                "cells": int(ok.sum()),
            }
        )
    return pd.DataFrame(rows)


# --- Heat trust -------------------------------------------------------------------------------


def heat_spread(values: pd.DataFrame, maps: dict) -> np.ndarray:
    """Range (max - min) of the corrected heat index across the three models, in degC."""
    from models import heat

    each = np.column_stack([heat.apply_map(maps[m], values[m]) for m in heat.MODELS])
    return each.max(axis=1) - each.min(axis=1)


def heat_trust(spread: np.ndarray, limits) -> np.ndarray:
    return np.where(spread < limits[0], 0, np.where(spread < limits[1], 1, 2))


def heat_evaluation() -> dict:
    """Spread limits on the heat model's fitting period; error and level misses by trust on
    the test year. Scored per ERA5 cell-day, as in Stage 9."""
    from models import heat

    model = heat.load_model()
    fitted = tuple(date.fromisoformat(d) for d in model["fitted_on"])
    climate = heat.era5_daily(*heat.CLIMATE)
    clim = heat.climatology(climate)
    recent = heat.era5_daily(fitted[0].year, HEAT_TEST[1].year)
    limits_rows = heat.table(fitted, recent, clim)
    limits = [
        round(float(np.percentile(heat_spread(limits_rows, model["maps"]), p)), 3)
        for p in TRUST_PERCENTILES
    ]
    rows = heat.table(HEAT_TEST, recent, clim)
    rows["value"] = heat.combine(rows, model["maps"])[model["method"]].values
    rows["spread"] = heat_spread(rows, model["maps"])
    cells = rows.groupby(["date", "cell"])[["value", "era5", "spread"]].mean().reset_index()
    cells["trust"] = heat_trust(cells["spread"].to_numpy(), limits)
    th = model["thresholds"]

    def lv(v):
        return np.searchsorted([th["yellow"], th["orange"], th["red"]], v, side="right")

    cells["level_miss"] = lv(cells["value"].to_numpy()) != lv(cells["era5"].to_numpy())
    table = []
    for code, level in enumerate(LEVELS):
        sel = cells[cells["trust"] == code]
        table.append(
            {
                "trust": level.value,
                "cases": len(sel),
                "mae": float((sel["value"] - sel["era5"]).abs().mean()) if len(sel) else np.nan,
                "level_miss": float(sel["level_miss"].mean()) if len(sel) else np.nan,
            }
        )
    return {"limits": limits, "fitted_on": model["fitted_on"], "table": pd.DataFrame(table)}


# --- Live use ---------------------------------------------------------------------------------


def save_model(params: dict, years: list[int]) -> dict:
    model = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fitted_on_years": years,
        "months": list(MONTHS),
        "models": MODELS,
        "rain": params,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=1), encoding="utf-8")
    load_model.cache_clear()
    return model


@lru_cache(maxsize=1)
def load_model() -> dict | None:
    """The saved blend, or None before Stage 11 has been run (callers then fall back)."""
    if not MODEL_PATH.exists():
        return None
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def from_api_models(fc: np.ndarray, names: list[str]) -> np.ndarray:
    """(model, lat, lon) in Open-Meteo model names -> (lat, lon, MODELS)."""
    return merge_models({n: fc[i] for i, n in enumerate(names) if n in API_MODELS})


def live_window(day: date) -> np.ndarray | None:
    """Rain per model (lat, lon, MODELS) for one IMD day from the clean Open-Meteo files."""
    parts = []
    for d in (day - timedelta(days=1), day):
        path = store.clean_path("openmeteo", d)
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            parts.append(ds["precipitation"].load())
    names = [str(m) for m in parts[0]["model"].values if m in parts[1]["model"].values]
    hourly = xr.concat([p.sel(model=names) for p in parts], dim="time")
    end = np.datetime64(day.isoformat()) + np.timedelta64(forecast.OBS_END_HOUR, "h")
    window = hourly.sel(time=slice(end - np.timedelta64(23, "h"), end))
    if window.sizes["time"] != 24:
        return None
    total = window.sum(dim="time", skipna=False).transpose("model", "lat", "lon").values
    return from_api_models(total, names)


def ward_trust(fields: list[np.ndarray]) -> pd.Series | None:
    """Trust per ward from the forecast windows used (each (lat, lon, MODELS)): the lowest
    over the windows. Empty list: every window was observed, so high. None without a model."""
    from models import flood

    model = load_model()
    points = flood.ward_points()
    if not fields:
        return pd.Series(Trust.HIGH, index=points.index)
    if model is None:
        return None
    lons, lats = target_coords()
    worst = np.zeros((lats.size, lons.size), dtype=int)
    for field in fields:
        worst = np.maximum(worst, predict(model["rain"], field)["trust"])
    grid = xr.DataArray(worst, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    lat = xr.DataArray(points["lat"].values, dims="ward")
    lon = xr.DataArray(points["lon"].values, dims="ward")
    codes = grid.sel(lat=lat, lon=lon, method="nearest").values
    return pd.Series([LEVELS[int(c)] for c in codes], index=points.index)


# --- Report -----------------------------------------------------------------------------------


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _n(v, digits: int = 2) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def _pair(p: tuple[int, int]) -> str:
    busted, cases = p
    return "-" if cases == 0 else f"{busted / cases:.0%} ({busted} of {cases})"


def plot_reliability(h: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(EVENTS), figsize=(9, 4.2), sharey=True)
    for ax, (amount, event) in zip(axes, EVENTS.items(), strict=True):
        observed = (h["obs"] >= amount).astype(float)
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect")
        for key, label, colour in (
            (f"share_{amount}", "share of models", "#d95f02"),
            (f"prob_{amount}", "blend probability", "#1b6ec2"),
        ):
            table = reliability(h[key], observed)
            table = table[table["cases"] >= MIN_BIN]
            ax.plot(table["forecast"], table["observed"], marker="o", color=colour, label=label)
            for _, r in table.iterrows():
                ax.annotate(
                    f"{r['cases']}",
                    (r["forecast"], r["observed"]),
                    textcoords="offset points",
                    xytext=(4, -10),
                    fontsize=6,
                    color=colour,
                )
        ax.set_title(f"{event.capitalize()} rain ({amount:g} mm or more)", fontsize=10)
        ax.set_xlabel("forecast probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("observed frequency (IMD)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Reliability on held-out monsoons {YEARS[0]}-{YEARS[-1]} (numbers: cell-days per bin; "
        f"bins under {MIN_BIN} not drawn)",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def report(h, seasons, forward, final, heat_eval, plot_name, command) -> str:
    methods = method_scores(h)
    by_trust = trust_table(h)
    t = by_trust.set_index("trust")
    blend_row, stage4_row = methods.iloc[2], methods.iloc[1]
    fwd = forward["methods"].set_index("method")
    fwd_trust = forward["trust"].set_index("trust")
    cells = h["obs"][0].size
    last = YEARS[-1]
    flagged = low_trust_days(h, last)
    amounts = trust_by_amount(h).set_index("forecast")
    heavy_row = amounts.loc["heavy or more"]
    years_table = per_year(h, seasons)
    low_above = sum(_rate(r["low"]) > _rate(r["high"]) for r in years_table.to_dict("records"))
    overlap = t.loc["low", "bust_range"][0] <= t.loc["high", "bust_range"][1]
    top = {
        key: reliability(h[f"{key}_64.5"], (h["obs"] >= 64.5).astype(float)).iloc[-1]
        for key in ("share", "prob")
    }
    lines = [
        "# Blending and bust detection: how far to trust each rain forecast",
        "",
        f"Generated by `{command}`. Do not edit by hand.",
        "",
        "## Result",
        "",
        f"- **Busts by trust, held-out monsoons {YEARS[0]}-{last}.** A bust (forecast and IMD two "
        f"or more rain classes apart) happened on {t.loc['high', 'bust_rate']:.1%} of high-trust "
        f"cell-days, {t.loc['medium', 'bust_rate']:.1%} of medium and "
        f"{t.loc['low', 'bust_rate']:.1%} of low. Low trust covers {t.loc['low', 'share']:.0%} of "
        f"cell-days and holds {t.loc['low', 'busts']} of the {int(by_trust['busts'].sum())} busts. "
        + (
            "The 90% ranges of high and low overlap, so overall this is a weak signal."
            if overlap
            else "The 90% ranges of high and low do not overlap."
        )
        + f" Low trust busted more often than high in {low_above} of {len(years_table)} years.",
        f"- **Heavy-rain forecasts.** When the blend forecast heavy rain, it busted on "
        f"{_pair(heavy_row['high'])} high-trust cell-days, {_pair(heavy_row['medium'])} medium "
        f"and {_pair(heavy_row['low'])} low. These are the forecasts an alert would rest on.",
        f"- **Forward check** (fit on {YEARS[0]}-{last - 1}, test {last} only): bust rate "
        f"{fwd_trust.loc['high', 'bust_rate']:.1%} high, "
        f"{fwd_trust.loc['medium', 'bust_rate']:.1%} medium, "
        f"{fwd_trust.loc['low', 'bust_rate']:.1%} low.",
        f"- **Blend against Stage 4's way** (one quantile map on the plain mean), same held-out "
        f"years: MAE {blend_row['mae']:.1f} against {stage4_row['mae']:.1f} mm, heavy-rain CSI "
        f"{_n(blend_row['heavy_csi'])} against {_n(stage4_row['heavy_csi'])}, moderate CSI "
        f"{_n(blend_row['moderate_csi'])} against {_n(stage4_row['moderate_csi'])}. "
        + _verdict(blend_row, stage4_row),
        f"- **Probabilities** (reliability plot below). Brier skill against climatology, heavy "
        f"rain: {_n(h['brier']['prob_64.5']['skill'])} for the blend probability, "
        f"{_n(h['brier']['share_64.5']['skill'])} for the plain share of models above 64.5 mm. "
        f"At the top, heavy rain followed {top['prob']['observed']:.0%} of the "
        f"{int(top['prob']['cases'])} blend forecasts in {top['prob']['bin']} and "
        f"{top['share']['observed']:.0%} of the {int(top['share']['cases'])} share-of-models "
        f"forecasts in {top['share']['bin']}.",
        f"- **Low-trust cases flagged:** {len(flagged)} days of {last} had at least half the "
        "cells at low trust; they are listed below. Risk runs now carry this trust per ward "
        "(map and API), and the API names the low-trust wards in its warnings.",
        *_heat_result(heat_eval),
        "",
        "## Setup",
        "",
        "- **Forecasts:** Open-Meteo's archive of stitched short-lead forecasts (about a day "
        "ahead), June-September, at the "
        f"{cells} pilot grid cells. Models present each year are in the per-year table; ECMWF "
        "is IFS 0.4 degree until February 2024 and IFS 0.25 degree after.",
        "- **Truth:** IMD gridded daily rain, the 24 hours ending 08:30 IST.",
        "- **Blend:** each model quantile-mapped to IMD on its own (training years where it "
        "exists), then the corrected models present averaged with weights 1 / MAE.",
        "- **Probability:** logistic regression of the event (IMD at or above 15.6 or 64.5 mm) "
        "on the share of corrected models at or above it and log(1 + blend).",
        f"- **Trust:** spread (standard deviation) of log(1 + corrected rain) across the models "
        f"present. High below the {TRUST_PERCENTILES[0]}th percentile of the training years' "
        f"wet cell-days (any model at 2.5 mm or more), low above the {TRUST_PERCENTILES[1]}th, "
        "medium between; every model dry = high; fewer than two models = low. Final limits: "
        f"{final['rain']['trust_limits'][0]:.2f} and {final['rain']['trust_limits'][1]:.2f}.",
        "- **Bust:** forecast and IMD two or more classes apart (dry < 2.5, rain < 15.6, "
        "moderate < 64.5, heavy < 115.6 mm, very heavy).",
        "- **Split:** leave one year out; each monsoon is scored by a fit on the other four, so "
        "no forecast is scored by a fit that saw its season. Neighbouring cells share IMD values "
        "and weather, so the cell-days are far from independent; the 90% ranges resample whole "
        "days.",
        f"- **Saved model** (`{MODEL_PATH.relative_to(settings.processed_dir.parent).as_posix()}`)"
        f": fitted on all of {YEARS[0]}-{last}; blend weights "
        + ", ".join(
            f"{LABELS[m]} {w / sum(final['rain']['weights'].values()):.2f}"
            for m, w in final["rain"]["weights"].items()
        )
        + ".",
        "",
        "## Busts by trust (held-out years)",
        "",
        _row(
            [
                "Trust",
                "Cell-days",
                "Share",
                "MAE (mm)",
                "Busts",
                "Bust rate",
                "90% range",
                "Heavy rain seen",
                "Heavy rain missed",
            ]
        ),
        _row(["---"] * 9),
    ]
    for r in by_trust.to_dict("records"):
        low, high = r["bust_range"]
        lines.append(
            _row(
                [
                    r["trust"],
                    r["cases"],
                    f"{r['share']:.0%}",
                    _n(r["mae"], 1),
                    r["busts"],
                    f"{r['bust_rate']:.1%}",
                    f"{low:.1%} to {high:.1%}",
                    r["heavy_seen"],
                    r["heavy_missed"],
                ]
            )
        )
    lines += [
        "",
        "### Bust rate by trust within each forecast amount",
        "",
        "Big forecasts disagree more and bust more, so trust could just repeat the amount. Within "
        "each class of blended rain, the bust rate should still rise from high to low trust.",
        "",
        _row(["Blended forecast", "High", "Medium", "Low"]),
        _row(["---"] * 4),
        *[
            _row([r["forecast"], _pair(r["high"]), _pair(r["medium"]), _pair(r["low"])])
            for r in trust_by_amount(h).to_dict("records")
        ],
        "",
        "### By year",
        "",
        "Bust rate by trust level, with the models the archive has that year.",
        "",
        _row(["Year", "Days", "Models", "High", "Medium", "Low"]),
        _row(["---"] * 6),
        *[
            _row(
                [
                    r["year"],
                    r["days"],
                    r["models"],
                    _pair(r["high"]),
                    _pair(r["medium"]),
                    _pair(r["low"]),
                ]
            )
            for r in per_year(h, seasons).to_dict("records")
        ],
        "",
        "## Reliability",
        "",
        f"![Reliability plot]({plot_name})",
        "",
        "A point on the diagonal means that when the forecast says p, the event happens a "
        "fraction p of the time. Points below it are overconfident.",
        "",
    ]
    for amount, event in EVENTS.items():
        observed = (h["obs"] >= amount).astype(float)
        share_t = reliability(h[f"share_{amount}"], observed).set_index("bin")
        prob_t = reliability(h[f"prob_{amount}"], observed).set_index("bin")
        lines += [
            f"### {event.capitalize()} rain ({amount:g} mm or more)",
            "",
            f"Brier score: blend probability {_n(h['brier'][f'prob_{amount}']['brier'], 3)} "
            f"(skill {_n(h['brier'][f'prob_{amount}']['skill'])}), share of models "
            f"{_n(h['brier'][f'share_{amount}']['brier'], 3)} (skill "
            f"{_n(h['brier'][f'share_{amount}']['skill'])}). Skill is against the training "
            "years' frequency of the event; 0 = no better, 1 = perfect.",
            "",
            _row(
                [
                    "Probability bin",
                    "Share of models: cases",
                    "observed",
                    "Blend: cases",
                    "observed",
                ]
            ),
            _row(["---"] * 5),
        ]
        for b in sorted(set(share_t.index) | set(prob_t.index)):
            s = share_t.loc[b] if b in share_t.index else None
            p = prob_t.loc[b] if b in prob_t.index else None
            lines.append(
                _row(
                    [
                        b,
                        int(s["cases"]) if s is not None else "-",
                        f"{s['observed']:.2f}" if s is not None else "-",
                        int(p["cases"]) if p is not None else "-",
                        f"{p['observed']:.2f}" if p is not None else "-",
                    ]
                )
            )
        lines.append("")
    lines += [
        "## Blend against simpler combinations (held-out years)",
        "",
        _row(
            [
                "Method",
                "MAE (mm)",
                "Moderate CSI",
                "Moderate 90% range",
                "Heavy CSI",
                "Heavy 90% range",
                "Heavy bias",
                "Bust rate",
            ]
        ),
        _row(["---"] * 8),
    ]
    for r in methods.to_dict("records"):
        lines.append(
            _row(
                [
                    r["method"],
                    _n(r["mae"], 1),
                    _n(r["moderate_csi"]),
                    f"{_n(r['moderate_range'][0])} to {_n(r['moderate_range'][1])}",
                    _n(r["heavy_csi"]),
                    f"{_n(r['heavy_range'][0])} to {_n(r['heavy_range'][1])}",
                    _n(r["heavy_bias"]),
                    f"{r['bust_rate']:.1%}",
                ]
            )
        )
    lines += [
        "",
        f"Forward check, {last} only (fit on {YEARS[0]}-{last - 1}):",
        "",
        _row(["Method", "MAE (mm)", "Moderate CSI", "Heavy CSI", "Bust rate"]),
        _row(["---"] * 5),
        *[
            _row(
                [
                    m,
                    _n(r["mae"], 1),
                    _n(r["moderate_csi"]),
                    _n(r["heavy_csi"]),
                    f"{r['bust_rate']:.1%}",
                ]
            )
            for m, r in fwd.iterrows()
        ],
        "",
        f"## Low-trust days in {last}",
        "",
        "Days with at least half the pilot cells at low trust. Rain is the city mean in mm for "
        "the 24 hours ending 08:30 IST; each model's amount is after its correction.",
        "",
    ]
    if flagged.empty:
        lines.append("None.")
    else:
        lines += [
            _row(
                ["Date", "Cells at low trust", "Models (corrected)", "Blend", "IMD", "Cells busted"]
            ),
            _row(["---"] * 6),
        ]
        for r in flagged.to_dict("records"):
            lines.append(
                _row(
                    [
                        r["date"].isoformat(),
                        f"{r['low_share']:.0%}",
                        r["models"],
                        f"{r['blend']:.0f}",
                        f"{r['imd']:.0f}",
                        f"{r['busts']} of {r['cells']}",
                    ]
                )
            )
    lines += ["", "## Heat: trust from the three models' spread", ""]
    if heat_eval is None:
        lines.append(
            "Not run: the Stage 9 heat model is missing (python -m models.heat --train ...)."
        )
    else:
        ht = heat_eval["table"]
        lines += [
            f"Spread = range of the three corrected apparent-temperature forecasts. Limits from "
            f"the heat model's fitting period ({heat_eval['fitted_on'][0]} to "
            f"{heat_eval['fitted_on'][1]}): high below {heat_eval['limits'][0]:.2f} °C, low from "
            f"{heat_eval['limits'][1]:.2f} °C. Scored on {HEAT_TEST[0]:%Y} per ERA5 cell-day; a "
            "level miss is a forecast level (green, yellow, orange, red) different from ERA5's.",
            "",
            _row(["Trust", "Cell-days", "MAE (°C)", "Level missed"]),
            _row(["---"] * 4),
            *[
                _row(
                    [
                        r["trust"],
                        r["cases"],
                        _n(r["mae"]),
                        f"{r['level_miss']:.0%}" if r["cases"] else "-",
                    ]
                )
                for r in ht.to_dict("records")
            ],
            "",
        ]
    lines += [
        "## Limits",
        "",
        "- About a day ahead only: the archive stitches short-lead forecasts. Disagreement grows "
        "with lead time; the limits here were not learned on longer leads (the Stage 7 replays "
        "apply them unchanged to forecasts made 1-3 days ahead with three models).",
        "- The model set changes over the years (two models in 2021, six from 2023), and the "
        "spread of two models is a noisier measure than the spread of six. Live runs fetch all "
        "six. With only ECMWF, GFS and ICON (forecast files fetched before Stage 11, and the "
        "Stage 7 replays) the spread is smaller, so fewer cells come out low.",
        "- IMD is itself a 0.25 degree interpolation of rain gauges, so part of a 'bust' at one "
        "cell can be IMD missing a downpour between gauges.",
        "- Trust is about the rain forecast. The flood model's own weakness (Stage 5: unreviewed "
        "labels, warns too often) is shown on the map as a separate caveat.",
        "",
        "## Next",
        "",
        "- Learn the limits by lead time from the Previous Runs archive (2024 onwards).",
        "- Use trust in the Stage 13 alert rules: no red alert on low trust without operator "
        "review.",
        "",
    ]
    return "\n".join(lines)


def _rate(p: tuple[int, int]) -> float:
    return p[0] / p[1] if p[1] else float("nan")


def _heat_result(heat_eval: dict | None) -> list[str]:
    if heat_eval is None:
        return []
    t = heat_eval["table"].set_index("trust")
    works = t.loc["low", "level_miss"] > t.loc["high", "level_miss"]
    return [
        "- **Heat.** The spread of the three heat forecasts "
        + (
            "does grade them"
            if works
            else "does not grade them: the level was missed more often at high trust"
        )
        + f" ({t.loc['high', 'level_miss']:.0%} high, {t.loc['medium', 'level_miss']:.0%} "
        f"medium, {t.loc['low', 'level_miss']:.0%} low, {HEAT_TEST[0]:%Y}). "
        + (
            "It is not yet used by heat runs."
            if works
            else "Heat runs keep their fixed medium trust until a measure that works is found."
        )
    ]


def _verdict(blend_row, stage4_row) -> str:
    better_mae = blend_row["mae"] <= stage4_row["mae"]
    better_heavy = blend_row["heavy_csi"] >= stage4_row["heavy_csi"]
    if better_mae and better_heavy:
        return "The blend is at least as good on both."
    if not better_mae and not better_heavy:
        return "The blend is worse on both, so flood risk keeps Stage 4's rain amount."
    return "Mixed, within the ranges, so flood risk keeps Stage 4's rain amount."


def evaluate(seasons: dict[int, dict]) -> tuple[dict, dict]:
    years = sorted(seasons)
    h = held_out(seasons, {y: [o for o in years if o != y] for y in years})
    h["brier"] = {}
    for amount in EVENTS:
        observed = (h["obs"] >= amount).astype(float)
        for key in (f"prob_{amount}", f"share_{amount}"):
            h["brier"][key] = brier(h[key], observed, h[f"clim_{amount}"])
    f = held_out(seasons, {years[-1]: years[:-1]})
    forward = {"methods": method_scores(f), "trust": trust_table(f)}
    return h, forward


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blend forecast models and grade their trust.")
    parser.add_argument("--fetch", nargs=2, type=int, metavar=("FIRST", "LAST"), help="years")
    parser.add_argument("--report", type=Path, help="fit, score and write a Markdown report")
    args = parser.parse_args(argv)
    if not args.fetch and not args.report:
        parser.error("give --fetch and/or --report")

    if args.fetch:
        for year in range(args.fetch[0], args.fetch[1] + 1):
            for month in MONTHS:
                path = fetch_month(year, month)
                print(f"{year}-{month:02d}  {path.relative_to(settings.raw_dir).as_posix()}")

    if args.report:
        seasons = {y: load_season(y) for y in YEARS}
        for y, s in seasons.items():
            print(f"{y}: {len(s['days'])} days")
        h, forward = evaluate(seasons)
        heat_eval = None
        try:
            heat_eval = heat_evaluation()
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"heat trust skipped: {exc}")
        fc, obs = stack(seasons, YEARS)
        final = save_model(fit(fc, obs), YEARS)
        plot = args.report.with_name(args.report.stem + "_reliability.svg")
        plot_reliability(h, plot)
        command = "python -m models.blend " + " ".join(argv if argv is not None else sys.argv[1:])
        text = report(h, seasons, forward, final, heat_eval, plot.name, command)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(
            trust_table(h)[["trust", "cases", "bust_rate", "mae"]].round(3).to_string(index=False)
        )
        print(f"Saved {MODEL_PATH}")
        print(f"Wrote {args.report} and {plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
