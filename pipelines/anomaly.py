"""Isolation Forest fault detector for hourly series, tested on injected faults.

Range checks (pipelines/qc.py) catch impossible values. This catches values that are possible
but wrong for the moment: a spike, a stuck (flat) reading, a slow drift.

There are no station feeds yet, so the test series are the hourly ECMWF forecast at each pilot
grid cell (temperature, humidity, wind) and the hourly sea level. Faults are synthetic. The
numbers say how well the detector finds these injected faults, not how it does on real sensors.
Rain is left out: it is zero most hours, so a flat line is normal and needs a different test.

python -m pipelines.anomaly --start 2025-06-15 --end 2025-08-31 --report docs/qc_report.md
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import IsolationForest

from pipelines import store

FEATURES = ["second_diff", "dev_from_6h_median", "hour_anomaly", "ref_dev", "ref_dev_24h", "std_6h"]
FEATURES.append("hours_unchanged")
TRAIN_FRACTION = 0.7  # split by time: the first 70% trains, the rest is the test period
FALSE_ALARM_TARGET = 0.01  # threshold = this quantile of the scores on clean training data
TIDE_LAG = 25  # hours; a tidal day is about 24 h 50 min
GUARD = 24  # hours after a fault during which rolling features still remember it
SIZES = (2.0, 4.0, 8.0)  # spike and drift size, in standard deviations of the series
FLAT_HOURS = (6, 12, 24)
DRIFT_HOURS = 48
EVENTS = 60  # injected events per fault type and size, per variable


@dataclass
class Series:
    """One variable: values shaped (time, series) and the reference each series is compared to."""

    name: str
    hours: np.ndarray  # hour of day for each time step
    values: np.ndarray
    reference: str  # "neighbours" or "lag"

    def ref(self, values: np.ndarray) -> np.ndarray:
        if self.reference == "neighbours":  # the other grid cells at the same hour
            return np.repeat(np.nanmedian(values, axis=1, keepdims=True), values.shape[1], axis=1)
        return pd.DataFrame(values).shift(TIDE_LAG).to_numpy()  # the same series one tide ago


def load(start: date, end: date, model: str = "ecmwf_ifs025") -> list[Series]:
    """Read the clean zone for a date range. Days with no file are skipped."""
    days = [start + timedelta(days=n) for n in range((end - start).days + 1)]

    def read(source: str) -> xr.Dataset | None:
        paths = [p for p in (store.clean_path(source, d) for d in days) if p.exists()]
        if not paths:
            return None
        parts = []
        for path in paths:
            with xr.open_dataset(path) as ds:
                parts.append(ds.load())
        return xr.concat(parts, dim="time").sortby("time")

    out = []
    forecast = read("openmeteo")
    if forecast is not None:
        cube = forecast.sel(model=model).stack(cell=("lat", "lon"))
        hours = cube["time"].dt.hour.values
        for name in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m"):
            values = cube[name].transpose("time", "cell").values.astype("float64")
            out.append(Series(name, hours, values, "neighbours"))
    tide = read("tide")
    if tide is not None:
        values = tide["sea_level"].values.astype("float64")[:, None]
        out.append(Series("sea_level", tide["time"].dt.hour.values, values, "lag"))
    return out


class FaultDetector:
    """Isolation Forest on features that describe how a reading sits in its recent history."""

    def __init__(self, series: Series, n_train: int, seed: int = 0):
        self.series = series
        train = series.values[:n_train]
        hours = series.hours[:n_train]
        self.climatology = np.stack([np.nanmean(train[hours == h], axis=0) for h in range(24)])
        self.sigma = float(np.nanstd(train - self.climatology[hours]))
        self.ref_bias = np.nanmean(train - series.ref(train), axis=0)
        self.forest = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
        rows = self.features(train).reshape(-1, len(FEATURES))
        self.forest.fit(rows)
        self.threshold = float(np.quantile(self.forest.score_samples(rows), FALSE_ALARM_TARGET))

    def features(self, values: np.ndarray) -> np.ndarray:
        """(time, series) -> (time, series, feature). Every feature uses only past values."""
        x = pd.DataFrame(values)
        hours = self.series.hours[: len(values)]
        ref_dev = x - self.series.ref(values) - self.ref_bias
        changed = (x != x.shift(1)).to_numpy()
        index = np.arange(len(values))[:, None]
        last_change = np.maximum.accumulate(np.where(changed, index, 0), axis=0)
        scaled = [
            x - 2 * x.shift(1) + x.shift(2),
            x - x.shift(1).rolling(6).median(),
            x - self.climatology[hours],
            ref_dev,
            ref_dev.rolling(24).mean(),
            x.rolling(6).std(),
        ]
        cube = [f.to_numpy() / self.sigma for f in scaled]
        cube.append(np.minimum(index - last_change, 24) / 24)
        return np.nan_to_num(np.stack(cube, axis=-1))

    def flag(self, values: np.ndarray, column: int | None = None) -> np.ndarray:
        """True where a reading looks faulty. `column` scores one series only (faster)."""
        feats = self.features(values)
        if column is not None:
            feats = feats[:, [column]]
        scores = self.forest.score_samples(feats.reshape(-1, len(FEATURES)))
        return (scores < self.threshold).reshape(feats.shape[:2])


def inject(values: np.ndarray, kind: str, start: int, hours: int, size: float) -> np.ndarray:
    """Return a copy of a 1-D series with one fault. `size` is in the units of the series."""
    out = values.copy()
    stop = start + hours
    if kind == "spike":
        out[start] += size
    elif kind == "flat":
        out[start:stop] = out[start - 1]
    elif kind == "drift":
        out[start:stop] += size * np.arange(1, hours + 1) / hours
    else:
        raise ValueError(kind)
    return out


def fault_cases() -> list[tuple[str, str, int, float]]:
    """(kind, label, hours, size in sigmas)"""
    cases = [("spike", f"{s:g} sigma", 1, s) for s in SIZES]
    cases += [("flat", f"{h} h", h, 0.0) for h in FLAT_HOURS]
    cases += [("drift", f"{s:g} sigma over {DRIFT_HOURS} h", DRIFT_HOURS, s) for s in SIZES]
    return cases


def evaluate(series: Series, seed: int = 0) -> dict:
    """Train on the first part, inject faults into the rest, count how many are caught."""
    n_time, n_series = series.values.shape
    n_train = int(n_time * TRAIN_FRACTION)
    detector = FaultDetector(series, n_train, seed)
    rng = np.random.default_rng(seed)

    clean_flags = detector.flag(series.values)[n_train:]
    rows = []
    for kind, label, hours, size in fault_cases():
        caught, delays = 0, []
        for _ in range(EVENTS):
            col = int(rng.integers(n_series))
            start = int(rng.integers(n_train + 1, n_time - hours))
            sign = rng.choice([-1.0, 1.0])
            faulty = series.values.copy()
            faulty[:, col] = inject(
                faulty[:, col], kind, start, hours, sign * size * detector.sigma
            )
            flags = detector.flag(faulty, column=col)[start : start + hours, 0]
            if flags.any():
                caught += 1
                delays.append(int(np.argmax(flags)))
        rows.append(
            {
                "variable": series.name,
                "fault": kind,
                "size": label,
                "events": EVENTS,
                "caught": caught,
                "rate": caught / EVENTS,
                "median_delay_h": float(np.median(delays)) if delays else float("nan"),
            }
        )
    return {
        "variable": series.name,
        "sigma": detector.sigma,
        "train_hours": n_train,
        "test_hours": n_time - n_train,
        "series": n_series,
        "false_alarm_rate": float(clean_flags.mean()),
        "rows": rows,
    }


def report(results: list[dict], start: date, end: date) -> str:
    table = pd.DataFrame([row for r in results for row in r["rows"]])
    overall = table.groupby("fault", sort=False)[["events", "caught"]].sum()
    lines = [
        "# QC report: fault detection on injected faults",
        "",
        f"Generated by `python -m pipelines.anomaly --start {start} --end {end}`. "
        "Do not edit by hand.",
        "",
        "## What was tested",
        "",
        "- **Detector:** Isolation Forest (200 trees) on seven features of each hourly reading: "
        "second difference, deviation from the median of the previous 6 hours, deviation from "
        "the usual value for that hour of day, deviation from a reference (the other grid cells "
        f"at the same hour; for sea level, the same series {TIDE_LAG} hours earlier), the 24-hour "
        "mean of that deviation, the 6-hour standard deviation, and hours since the value last "
        "changed. Every feature uses past values only.",
        "- **Data:** hourly ECMWF forecast at the 30 pilot grid cells and hourly sea level, "
        f"{start} to {end}. These stand in for station feeds, which the project does not have "
        "yet. Rain is not tested: it is zero most hours, so a flat line is normal.",
        f"- **Split:** by time. The first {TRAIN_FRACTION:.0%} of hours train the detector; "
        "faults are injected only into the rest.",
        f"- **Threshold:** set so that {FALSE_ALARM_TARGET:.0%} of clean training readings "
        "are flagged.",
        f"- **Faults:** {EVENTS} events per row, each at a random series and hour, random sign. "
        "Sizes are in standard deviations (sigma) of the variable around its usual daily cycle. "
        "A spike is one reading; a flat line repeats the previous reading; a drift is a straight "
        f"ramp reaching the stated size after {DRIFT_HOURS} hours.",
        "- **Caught** means at least one reading inside the fault was flagged. Delay is the "
        "median number of hours from the start of the fault to the first flag.",
        "",
        "## Detection rate by fault type (all variables)",
        "",
        "| Fault | Events | Caught | Detection rate |",
        "|---|---|---|---|",
    ]
    for kind, row in overall.iterrows():
        rate = row["caught"] / row["events"]
        lines.append(f"| {kind} | {row['events']} | {row['caught']} | {rate:.0%} |")
    lines += [
        "",
        "## False alarms on clean test data",
        "",
        "| Variable | Sigma | Series | Train hours | Test hours | Readings flagged |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['variable']} | {r['sigma']:.2f} | {r['series']} | {r['train_hours']} | "
            f"{r['test_hours']} | {r['false_alarm_rate']:.1%} |"
        )
    lines += [
        "",
        "## Detection rate by variable, fault and size",
        "",
        "| Variable | Fault | Size | Events | Caught | Detection rate | Median delay (h) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table.to_dict("records"):
        delay = "" if np.isnan(row["median_delay_h"]) else f"{row['median_delay_h']:g}"
        lines.append(
            f"| {row['variable']} | {row['fault']} | {row['size']} | {row['events']} | "
            f"{row['caught']} | {row['rate']:.0%} | {delay} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        "- The faults are synthetic and the series are model output, which is smoother than a "
        "real sensor. Expect lower detection and more false alarms on real station data.",
        "- The test period is a few weeks of one monsoon season.",
        "- A flagged reading is only marked as suspect. Nothing is deleted.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--report", type=Path, help="write the Markdown report here")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    series = load(args.start, args.end)
    if not series:
        print("No clean openmeteo or tide files in that range. Run pipelines.run first.")
        return 1
    results = []
    for s in series:
        if len(s.values) < 24 * 14:
            print(f"{s.name}: only {len(s.values)} hours, need at least two weeks")
            return 1
        result = evaluate(s, args.seed)
        results.append(result)
        print(f"{s.name}: false alarms {result['false_alarm_rate']:.1%} of clean readings")
        for row in result["rows"]:
            print(f"  {row['fault']:<6} {row['size']:<20} {row['caught']}/{row['events']}")
    text = report(results, args.start, args.end)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
