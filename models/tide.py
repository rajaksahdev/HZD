"""Astronomical tide at Mumbai harbour for any date, from a harmonic fit.

Why: the flood model needs the high tide on each labelled day, and the labels go back to 2005.
Open-Meteo's hourly sea level (pipelines/sources/tide.py) only starts in 2023. The tide is a sum
of waves with known periods, so a fit on 2023-2024 can predict any past or future date.

How: least-squares fit of a mean level plus 18 tidal constituents, each scaled by its standard
18.6-year nodal correction (Schureman's formulas). What is fitted is Open-Meteo's sea level, a
model estimate rather than a gauge reading. The prediction holds the regular tide only: storm
surge and other weather-driven changes are left out.

python -m models.tide --fit 2023-01-01 2024-12-31 --check 2025-01-01 2025-10-31
"""

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from api.config import settings
from pipelines import store
from pipelines.sources import tide as tide_source

MODEL_PATH = settings.processed_dir / "models" / "tide_harmonics.json"
RAW = settings.raw_dir / tide_source.NAME / "yearly"
EPOCH = np.datetime64("2000-01-01T00:00")
IST = timedelta(hours=5, minutes=30)

# Constituent speeds in degrees per hour.
SPEEDS = {
    "Sa": 0.0410686,
    "Ssa": 0.0821373,
    "Mm": 0.5443747,
    "Mf": 1.0980331,
    "Q1": 13.3986609,
    "O1": 13.9430356,
    "P1": 14.9589314,
    "K1": 15.0410686,
    "J1": 15.5854433,
    "2N2": 27.8953548,
    "MU2": 27.9682084,
    "N2": 28.4397295,
    "NU2": 28.5125831,
    "M2": 28.9841042,
    "L2": 29.5284789,
    "S2": 30.0,
    "K2": 30.0821373,
    "M4": 57.9682084,
    "MS4": 58.9841042,
}


def _node_longitude(hours: np.ndarray) -> np.ndarray:
    """Longitude of the Moon's ascending node in radians; it turns once in 18.61 years."""
    days_since_1900 = hours / 24 + (EPOCH - np.datetime64("1899-12-31T12:00")) / np.timedelta64(
        1, "D"
    )
    return np.radians((259.1560564 - 0.0529539222 * days_since_1900) % 360)


def nodal(name: str, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude factor f and phase shift u (radians) for one constituent."""
    n = _node_longitude(hours)
    c1, c2, c3 = np.cos(n), np.cos(2 * n), np.cos(3 * n)
    s1, s2, s3 = np.sin(n), np.sin(2 * n), np.sin(3 * n)
    m2 = (1.0004 - 0.0373 * c1 + 0.0002 * c2, -2.14 * s1)
    o1 = (1.0089 + 0.1871 * c1 - 0.0147 * c2 + 0.0014 * c3, 10.80 * s1 - 1.34 * s2 + 0.19 * s3)
    k1 = (1.0060 + 0.1150 * c1 - 0.0088 * c2 + 0.0006 * c3, -8.86 * s1 + 0.68 * s2 - 0.07 * s3)
    table = {
        "M2": m2,
        "N2": m2,
        "2N2": m2,
        "MU2": m2,
        "NU2": m2,
        "L2": m2,
        "MS4": m2,
        "M4": (m2[0] ** 2, 2 * m2[1]),
        "O1": o1,
        "Q1": o1,
        "K1": k1,
        "J1": (
            1.0129 + 0.1676 * c1 - 0.0170 * c2 + 0.0016 * c3,
            -12.94 * s1 + 1.34 * s2 - 0.19 * s3,
        ),
        "K2": (
            1.0241 + 0.2863 * c1 + 0.0083 * c2 - 0.0015 * c3,
            -17.74 * s1 + 0.68 * s2 - 0.04 * s3,
        ),
        "Mm": (1.0 - 0.1300 * c1 + 0.0013 * c2, 0.0 * s1),
        "Mf": (1.0429 + 0.4135 * c1 - 0.004 * c2, -23.74 * s1 + 2.68 * s2 - 0.38 * s3),
    }
    f, u = table.get(name, (np.ones_like(n), np.zeros_like(n)))  # Sa, Ssa, P1, S2: no correction
    return np.broadcast_to(f, n.shape), np.radians(np.broadcast_to(u, n.shape))


def _hours(times) -> np.ndarray:
    times = np.asarray(times, dtype="datetime64[m]")
    return (times - EPOCH) / np.timedelta64(1, "h")


def _design(hours: np.ndarray, names: list[str]) -> np.ndarray:
    columns = [np.ones_like(hours)]
    for name in names:
        f, u = nodal(name, hours)
        angle = np.radians(SPEEDS[name]) * hours + u
        columns += [f * np.cos(angle), f * np.sin(angle)]
    return np.column_stack(columns)


def fit(times, heights, names: list[str] | None = None) -> dict:
    """Fit mean level and constituents to an hourly series. Missing heights are skipped."""
    names = names or list(SPEEDS)
    heights = np.asarray(heights, dtype=float)
    hours = _hours(times)
    ok = np.isfinite(heights)
    coef, *_ = np.linalg.lstsq(_design(hours[ok], names), heights[ok], rcond=None)
    constituents = {}
    for i, name in enumerate(names):
        a, b = coef[1 + 2 * i], coef[2 + 2 * i]
        constituents[name] = {
            "amplitude_m": float(np.hypot(a, b)),
            "phase_deg": float(np.degrees(np.arctan2(b, a)) % 360),
        }
    return {"mean_m": float(coef[0]), "constituents": constituents, "epoch": str(EPOCH)}


def predict(model: dict, times) -> np.ndarray:
    """Tide height (m above mean sea level) at the given UTC times."""
    hours = _hours(times)
    out = np.full(hours.shape, model["mean_m"], dtype=float)
    for name, c in model["constituents"].items():
        f, u = nodal(name, hours)
        angle = np.radians(SPEEDS[name]) * hours + u - np.radians(c["phase_deg"])
        out += f * c["amplitude_m"] * np.cos(angle)
    return out


def daily_high(model: dict, days) -> pd.Series:
    """Highest tide (m) in each local (IST) calendar day, on a 10-minute step."""
    out = {}
    for day in days:
        start = np.datetime64(datetime.combine(day, datetime.min.time()) - IST, "m")
        steps = start + np.arange(0, 24 * 60, 10).astype("timedelta64[m]")
        out[day] = float(predict(model, steps).max())
    return pd.Series(out, name="tide_high_m")


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m models.tide --fit ...")
    return json.loads(path.read_text(encoding="utf-8"))


# --- Data: one Open-Meteo request per calendar year, saved in the raw zone ---------------------


def _year_file(year: int) -> Path | None:
    files = sorted((RAW / str(year)).glob("sea_level_year_*.json"))
    return files[-1] if files else None


def fetch_year(year: int, force: bool = False) -> Path:
    today = datetime.now(UTC).date()
    have = _year_file(year)
    if have and not force and year < today.year:
        return have
    end = min(date(year, 12, 31), today - timedelta(days=1))
    resp = store.http_get(
        tide_source.URL,
        params={
            "latitude": tide_source.LAT,
            "longitude": tide_source.LON,
            "hourly": tide_source.VAR,
            "start_date": f"{year}-01-01",
            "end_date": end.isoformat(),
            "timezone": "UTC",
        },
        timeout=120,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RAW / str(year) / f"sea_level_year_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    store.log_download(tide_source.NAME, date(year, 1, 1), path, resp.url)
    return path


def observed(start: date, end: date) -> pd.Series:
    """Hourly Open-Meteo sea level (m) between two dates, downloading years as needed."""
    parts = []
    for year in range(start.year, end.year + 1):
        hourly = json.loads(fetch_year(year).read_text(encoding="utf-8"))["hourly"]
        values = [np.nan if v is None else v for v in hourly[tide_source.VAR]]
        parts.append(pd.Series(values, index=pd.to_datetime(hourly["time"]), dtype=float))
    series = pd.concat(parts).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series[str(start) : str(end)]


def check(model: dict, obs: pd.Series) -> dict:
    """How well the prediction matches Open-Meteo on a period it was not fitted on."""
    obs = obs.dropna()
    pred = pd.Series(predict(model, obs.index.values), index=obs.index)
    diff = pred - obs
    ist_day = (obs.index + IST).date
    highs = pd.DataFrame({"obs": obs.groupby(ist_day).max(), "pred": pred.groupby(ist_day).max()})
    return {
        "hours": int(len(obs)),
        "rmse_m": float(np.sqrt((diff**2).mean())),
        "correlation": float(np.corrcoef(pred, obs)[0, 1]),
        "obs_range_m": float(obs.max() - obs.min()),
        "daily_high_mae_m": float((highs["pred"] - highs["obs"]).abs().mean()),
        "days": int(len(highs)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit and check the harmonic tide model.")
    parser.add_argument("--fit", nargs=2, type=date.fromisoformat, metavar=("START", "END"))
    parser.add_argument("--check", nargs=2, type=date.fromisoformat, metavar=("START", "END"))
    args = parser.parse_args(argv)
    if not args.fit:
        parser.error("give --fit START END")
    if args.check and args.check[0] <= args.fit[1]:
        parser.error("the check period must start after the fit period ends")

    series = observed(*args.fit)
    model = fit(series.index.values, series.values)
    model["fitted_on"] = [d.isoformat() for d in args.fit]
    model["fitted_hours"] = int(series.notna().sum())
    if args.check:
        model["check"] = {"period": [d.isoformat() for d in args.check]}
        model["check"].update(check(model, observed(*args.check)))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(f"Fitted on {model['fitted_hours']} hours; saved {MODEL_PATH}")
    biggest = sorted(model["constituents"].items(), key=lambda kv: -kv[1]["amplitude_m"])[:6]
    print("Largest constituents: " + ", ".join(f"{k} {v['amplitude_m']:.2f} m" for k, v in biggest))
    if args.check:
        c = model["check"]
        print(
            f"Check {c['period'][0]} to {c['period'][1]} ({c['hours']} hours): "
            f"RMSE {c['rmse_m']:.3f} m, correlation {c['correlation']:.3f}, "
            f"daily high tide error {c['daily_high_mae_m']:.3f} m "
            f"(observed range {c['obs_range_m']:.2f} m)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
