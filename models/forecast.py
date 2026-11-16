"""Forecast core: daily rain from three public models, bias-corrected, scored against baselines.

What is compared, for each pilot grid cell and day:
  each model alone      ECMWF, GFS, ICON (from Open-Meteo)
  multi-model mean      the plain average of the three
  corrected mean        the average after quantile mapping (see below)
  persistence           tomorrow will be like the last observed day
Truth is the IMD gridded daily rainfall: the 24 hours ending 08:30 IST (03:00 UTC) on the date.

Quantile mapping: learn, on the training period, which observed amount sits at the same rank as
each forecast amount, and replace the forecast with it. It fixes "the model rains too often and
too lightly" but cannot fix a forecast that puts the rain on the wrong day.

Split by time: the mapping is learned on the training period and every score is from the test
period.

python -m models.forecast --train 2024-06-01 2024-09-30 --test 2025-06-01 2025-09-30 \
    --report docs/forecast_skill.md
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from api.config import settings
from pipelines import store

MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
# IMD daily rainfall classes: rain, moderate, heavy, very heavy (mm in 24 h).
THRESHOLDS = [2.5, 15.6, 64.5, 115.6]
QUANTILES = np.concatenate([np.linspace(0, 0.9, 19), np.linspace(0.91, 1, 10)])
BOOTSTRAP = 500
OBS_END_HOUR = 3  # the IMD day ends at 03:00 UTC


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


def observed(day: date) -> np.ndarray | None:
    """IMD rain (lat, lon) for the 24 h ending 03:00 UTC on `day`."""
    path = store.clean_path("imd", day)
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        return ds["precipitation"].values[0].astype("float64")


def forecast(day: date) -> np.ndarray | None:
    """Each model's rain (model, lat, lon) summed over the same 24 h as the IMD day.

    Open-Meteo's hourly value at time t is the rain in the hour ending at t, so the window is
    04:00 UTC on the previous date to 03:00 UTC on `day`.
    """
    parts = []
    for d in (day - timedelta(days=1), day):
        path = store.clean_path("openmeteo", d)
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            parts.append(ds["precipitation"].sel(model=MODELS).load())
    hourly = xr.concat(parts, dim="time")
    start = np.datetime64(day.isoformat()) - np.timedelta64(24 - OBS_END_HOUR - 1, "h")
    stop = np.datetime64(day.isoformat()) + np.timedelta64(OBS_END_HOUR, "h")
    window = hourly.sel(time=slice(start, stop))
    if window.sizes["time"] != 24:
        return None
    return window.sum(dim="time", skipna=False).values.astype("float64")


def load(start: date, end: date) -> dict:
    """Matched arrays for a period: obs (day, lat, lon), fc (day, model, lat, lon), persistence."""
    days, obs, fc, previous = [], [], [], []
    for day in _days(start, end):
        o, f, p = observed(day), forecast(day), observed(day - timedelta(days=1))
        if o is None or f is None or p is None:
            continue
        days.append(day)
        obs.append(o)
        fc.append(f)
        previous.append(p)
    if not days:
        raise RuntimeError(f"No matched forecast and IMD days between {start} and {end}")
    return {
        "days": days,
        "obs": np.stack(obs),
        "fc": np.stack(fc),
        "persistence": np.stack(previous),
        "expected_days": (end - start).days + 1,
    }


def fit_quantile_map(forecasts: np.ndarray, obs: np.ndarray) -> dict:
    """Paired quantiles of forecast and observed amounts, pooled over days and cells."""
    ok = np.isfinite(forecasts) & np.isfinite(obs)
    return {
        "forecast": np.quantile(forecasts[ok], QUANTILES).tolist(),
        "observed": np.quantile(obs[ok], QUANTILES).tolist(),
    }


def apply_quantile_map(mapping: dict, forecasts: np.ndarray) -> np.ndarray:
    """Replace each forecast amount with the observed amount at the same rank.

    Above the largest training forecast, the amount is scaled by the ratio at the top quantile.
    """
    fq, oq = np.asarray(mapping["forecast"]), np.asarray(mapping["observed"])
    fq = np.maximum.accumulate(fq + np.arange(fq.size) * 1e-9)  # strictly increasing for interp
    out = np.interp(forecasts, fq, oq)
    above = forecasts > fq[-1]
    if fq[-1] > 0:
        out = np.where(above, forecasts * oq[-1] / fq[-1], out)
    return np.where(np.isfinite(forecasts), out, np.nan)


def contingency(fc: np.ndarray, obs: np.ndarray, threshold: float) -> tuple[int, int, int]:
    """(hits, misses, false alarms) for the event 'rain >= threshold'."""
    ok = np.isfinite(fc) & np.isfinite(obs)
    f, o = fc[ok] >= threshold, obs[ok] >= threshold
    return int((f & o).sum()), int((~f & o).sum()), int((f & ~o).sum())


def scores(hits: int, misses: int, false_alarms: int) -> dict:
    """POD, FAR, CSI and frequency bias. NaN where the score is undefined."""

    def ratio(a, b):
        return a / b if b else float("nan")

    return {
        "pod": ratio(hits, hits + misses),
        "far": ratio(false_alarms, hits + false_alarms),
        "csi": ratio(hits, hits + misses + false_alarms),
        "bias": ratio(hits + false_alarms, hits + misses),
    }


def csi_interval(fc: np.ndarray, obs: np.ndarray, threshold: float, seed: int = 0):
    """90% interval for CSI from resampling whole days (cells on one day are not independent)."""
    rng = np.random.default_rng(seed)
    n = fc.shape[0]
    per_day = np.array([contingency(fc[i], obs[i], threshold) for i in range(n)])
    picks = rng.integers(0, n, size=(BOOTSTRAP, n))
    totals = per_day[picks].sum(axis=1)
    denominator = totals.sum(axis=1)
    csi = np.where(denominator > 0, totals[:, 0] / np.maximum(denominator, 1), np.nan)
    if np.isnan(csi).all():
        return float("nan"), float("nan")
    return float(np.nanquantile(csi, 0.05)), float(np.nanquantile(csi, 0.95))


def evaluate(train: dict, test: dict) -> tuple[pd.DataFrame, dict]:
    mapping = fit_quantile_map(train["fc"].mean(axis=1), train["obs"])
    mean = test["fc"].mean(axis=1)
    methods = {MODELS[m]: test["fc"][:, m] for m in range(len(MODELS))}
    methods["multi-model mean"] = mean
    methods["corrected mean"] = apply_quantile_map(mapping, mean)
    methods["persistence"] = test["persistence"]

    rows = []
    for threshold in THRESHOLDS:
        events = int((test["obs"] >= threshold).sum())
        for name, fc in methods.items():
            hits, misses, false_alarms = contingency(fc, test["obs"], threshold)
            low, high = csi_interval(fc, test["obs"], threshold)
            rows.append(
                {
                    "threshold_mm": threshold,
                    "method": name,
                    "observed_events": events,
                    "hits": hits,
                    "misses": misses,
                    "false_alarms": false_alarms,
                    **scores(hits, misses, false_alarms),
                    "csi_low": low,
                    "csi_high": high,
                    "mae_mm": float(np.nanmean(np.abs(fc - test["obs"]))),
                }
            )
    return pd.DataFrame(rows), mapping


def _fmt(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2f}"


def report(table: pd.DataFrame, train: dict, test: dict, args) -> str:
    cells = test["obs"][0].size
    labels = {2.5: "rain", 15.6: "moderate", 64.5: "heavy", 115.6: "very heavy"}
    lines = [
        "# Forecast skill: daily rain against baselines",
        "",
        f"Generated by `python -m models.forecast --train {args.train[0]} {args.train[1]} "
        f"--test {args.test[0]} {args.test[1]}`. Do not edit by hand.",
        "",
        "## What was tested",
        "",
        "- **Forecasts:** daily rain from ECMWF, GFS and ICON (Open-Meteo), their plain average, "
        "and the average after quantile mapping learned on the training period.",
        "- **Lead time:** past days come from Open-Meteo's archive of stitched short-lead "
        "forecasts, so these scores are for forecasts issued up to about a day ahead. Skill at "
        "two or more days ahead is not measured here and will be lower.",
        "- **Truth:** IMD gridded daily rainfall (0.25 degree), the 24 hours ending 08:30 IST, "
        f"interpolated to the {cells} pilot grid cells. Neighbouring cells share the same IMD "
        "values, so the cells are far from independent.",
        "- **Baseline:** persistence, which repeats the last observed day.",
        f"- **Split by time:** trained on {train['days'][0]} to {train['days'][-1]} "
        f"({len(train['days'])} of {train['expected_days']} days matched), tested on "
        f"{test['days'][0]} to {test['days'][-1]} ({len(test['days'])} of "
        f"{test['expected_days']} days matched).",
        "- **Scores:** POD = share of observed events that were forecast. FAR = share of "
        "forecast events that did not happen. CSI = hits / (hits + misses + false alarms). "
        "Bias = forecast events / observed events. The CSI range is a 90% interval from "
        "resampling whole days.",
        "- **Not done:** the pySTEPS nowcast baseline needs a sequence of recent rain maps "
        "(IMERG or GSMaP), which the project cannot download until the accounts exist. The "
        "ConvLSTM experiment is separate work on Kaggle.",
        "",
    ]
    for threshold in THRESHOLDS:
        part = table[table["threshold_mm"] == threshold]
        events = int(part["observed_events"].iloc[0])
        total = int(np.isfinite(test["obs"]).sum())
        lines += [
            f"## {labels[threshold].capitalize()}: {threshold:g} mm or more in a day",
            "",
            f"Observed in {events} of {total} cell-days.",
            "",
            "| Method | Hits | Misses | False alarms | POD | FAR | CSI | CSI 90% range | Bias |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in part.to_dict("records"):
            interval = f"{_fmt(row['csi_low'])} to {_fmt(row['csi_high'])}"
            lines.append(
                f"| {row['method']} | {row['hits']} | {row['misses']} | {row['false_alarms']} | "
                f"{_fmt(row['pod'])} | {_fmt(row['far'])} | {_fmt(row['csi'])} | {interval} | "
                f"{_fmt(row['bias'])} |"
            )
        lines.append("")
    mae = table.drop_duplicates("method")[["method", "mae_mm"]]
    lines += ["## Mean absolute error (mm per day)", "", "| Method | MAE |", "|---|---|"]
    lines += [f"| {r['method']} | {r['mae_mm']:.1f} |" for r in mae.to_dict("records")]
    lines.append("")
    return "\n".join(lines)


def mapping_path() -> Path:
    return settings.processed_dir / "models" / "rain_quantile_map.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score daily rain forecasts against baselines.")
    parser.add_argument("--train", nargs=2, type=date.fromisoformat, required=True)
    parser.add_argument("--test", nargs=2, type=date.fromisoformat, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.train[1] >= args.test[0]:
        parser.error("the training period must end before the test period starts")

    train, test = load(*args.train), load(*args.test)
    table, mapping = evaluate(train, test)
    path = mapping_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {"trained_on": [d.isoformat() for d in args.train], "models": MODELS, **mapping}
    path.write_text(json.dumps(saved, indent=1), encoding="utf-8")

    shown = table[["threshold_mm", "method", "hits", "misses", "false_alarms", "pod", "far", "csi"]]
    print(shown.round(2).to_string(index=False))
    print(f"\nQuantile map saved to {path}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report(table, train, test, args), encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
