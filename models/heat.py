"""Heat stress (Stage 9): a heat level per ward from the apparent temperature.

Index      Apparent temperature (Steadman 1994, the shade version used by the Australian Bureau
           of Meteorology): AT = T + 0.33 e - 0.70 ws - 4.00, with T the air temperature (degC),
           e the water vapour pressure (hPa) and ws the 10 m wind speed (m/s). Humid air raises
           it and wind lowers it; direct sun would add more. The daily index is the highest
           hourly value in the local (IST) day.
Levels     Percentiles of the daily index over Mumbai in ERA5, 1991-2020: yellow from the 90th,
           orange from the 95th, red from the 99th. Local percentiles, because what counts as
           dangerous heat depends on what people are used to. Provisional until Stage 13.
ERA5       Hourly reanalysis (0.25 degree) through Open-Meteo's historical weather API, no
           account needed. Over Mumbai the API serves two land cells, both at 73.00 E; each ward
           takes the nearer. ERA5 therefore gives city-scale heat and cannot see differences
           between wards.
Forecast   Each model's (ECMWF, GFS, ICON) daily index at each ward's point, quantile-mapped to
           ERA5. Three ways to combine them are compared on a validation period and the best is
           kept: the corrected mean, the corrected median (two of three models agree) and
           corrected ECMWF alone. Then the correction is refitted on the training and
           validation periods and scored once on a later test period (split by time). The
           ECMWF archive starts in February 2024, which sets the earliest training day.
Probability  Share of the three corrected models at orange or above.

python -m models.heat --fetch 1991 2025                            ERA5, one request per year
python -m models.heat --train 2024-02-01 2024-07-31 --validate 2024-08-01 2024-12-31 \
    --test 2025-01-01 2025-12-31 --report docs/heat.md
python -m models.heat --date today                                 heat per ward -> PostGIS
python -m models.heat --date 2025-04-15 --source era5              a past day from ERA5 (TEST)
"""

import argparse
import json
import sys
import time
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from api.config import settings
from api.schemas import Hazard, Severity, Trust, WardRisk
from models import flood, risk
from pipelines import store

NAME = "era5"
URL = "https://archive-api.open-meteo.com/v1/archive"
RAW = settings.raw_dir / NAME
CELLS = {"19.00N": (19.0, 73.0), "19.25N": (19.25, 73.0)}  # the land cells the API serves
VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
CLIMATE = (1991, 2020)
PERCENTILES = {Severity.YELLOW: 90, Severity.ORANGE: 95, Severity.RED: 99}
QUANTILES = np.linspace(0, 1, 41)
ERA5_DELAY = timedelta(days=6)  # ERA5 reaches the API about five days after real time
MODEL_PATH = settings.processed_dir / "models" / "heat.json"
UNIT = "°C"
SOURCES = ("forecast", "era5")
CANDIDATES = {  # ways to combine the corrected models; one is chosen on the validation year
    "corrected_mean": "corrected mean",
    "corrected_median": "corrected median (2 of 3 models)",
    "corrected_ecmwf": "corrected ECMWF",
}
REFERENCE = {
    "mean": "multi-model mean, uncorrected",
    "ecmwf_ifs025": "ECMWF, uncorrected",
    "gfs_seamless": "GFS, uncorrected",
    "icon_seamless": "ICON, uncorrected",
    "persistence": "persistence (ERA5 yesterday)",
    "climatology": "climatology (1991-2020)",
}
# Day-ahead scores are in docs/heat.md, but levels are city-scale and the thresholds are not
# yet tied to health outcomes. Stage 11 replaces this with a trust computed from disagreement.
TRUST = Trust.MEDIUM


def apparent_temperature(t, rh, ws):
    """Steadman apparent temperature (degC) without radiation. Works on arrays and DataArrays."""
    vapour = rh / 100 * 6.105 * np.exp(17.27 * t / (237.7 + t))
    return t + 0.33 * vapour - 0.70 * ws - 4.00


def level(value: float, thresholds: dict) -> Severity:
    for name in (Severity.RED, Severity.ORANGE, Severity.YELLOW):
        if value >= thresholds[name.value]:
            return name
    return Severity.GREEN


def ward_cells() -> pd.Series:
    """The ERA5 cell each ward takes: the nearer of the two by latitude."""
    lats = flood.ward_points()["lat"]
    names, cell_lats = list(CELLS), np.array([c[0] for c in CELLS.values()])
    return lats.map(lambda lat: names[int(np.abs(cell_lats - lat).argmin())])


# --- ERA5 -----------------------------------------------------------------------------------


def _year_file(year: int) -> Path | None:
    files = sorted((RAW / f"{year}").glob("era5_*.json"))
    return files[-1] if files else None


def fetch_year(year: int, force: bool = False) -> Path:
    """Hourly ERA5 for both cells for one year, saved raw. A finished year is fetched once."""
    have = _year_file(year)
    last = min(date(year, 12, 31), datetime.now(UTC).date() - ERA5_DELAY)
    if have and last == date(year, 12, 31) and not force:
        return have
    resp = store.http_get(
        URL,
        params={
            "latitude": ",".join(f"{c[0]:.2f}" for c in CELLS.values()),
            "longitude": ",".join(f"{c[1]:.2f}" for c in CELLS.values()),
            "hourly": ",".join(VARIABLES),
            "models": "era5",
            "wind_speed_unit": "ms",
            "start_date": f"{year}-01-01",
            "end_date": last.isoformat(),
            "timezone": "UTC",
        },
        timeout=300,
        retries=6,
    )
    got = [(round(loc["latitude"], 2), round(loc["longitude"], 2)) for loc in resp.json()]
    if got != list(CELLS.values()):
        raise RuntimeError(f"ERA5 cells changed: asked for {list(CELLS.values())}, got {got}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RAW / f"{year}" / f"era5_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    store.log_download(NAME, date(year, 1, 1), path, resp.url)
    return path


@lru_cache(maxsize=64)
def era5_year(year: int) -> pd.DataFrame | None:
    """Hourly apparent temperature (UTC, naive) for one year, one column per cell."""
    path = _year_file(year)
    if path is None:
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    columns = {}
    for name, loc in zip(CELLS, body, strict=True):
        h = loc["hourly"]
        t, rh, ws = (np.array(h[v], dtype=float) for v in VARIABLES)  # null -> NaN
        columns[name] = pd.Series(apparent_temperature(t, rh, ws), index=pd.to_datetime(h["time"]))
    return pd.DataFrame(columns)


def daily_max(hourly: pd.DataFrame) -> pd.DataFrame:
    """Highest hourly value in each IST day; NaN unless all 24 hours are there."""
    day = (hourly.index + pd.Timedelta(hours=5, minutes=30)).normalize()
    grouped = hourly.groupby(day)
    out = grouped.max().where(grouped.count() == 24)
    out.index = out.index.date
    return out


def era5_daily(first: int, last: int) -> pd.DataFrame:
    """Daily index per cell for whole years (the year before supplies the first IST hours)."""
    parts = [h for y in range(first - 1, last + 1) if (h := era5_year(y)) is not None]
    if not parts:
        raise FileNotFoundError(f"No ERA5 in {RAW}. Run: python -m models.heat --fetch ...")
    hourly = pd.concat(parts).sort_index()
    hourly = hourly[~hourly.index.duplicated()]
    daily = daily_max(hourly)
    years = np.array([d.year for d in daily.index])
    return daily[(years >= first) & (years <= last)].dropna(how="all")


def thresholds(daily: pd.DataFrame) -> dict[str, float]:
    values = daily.to_numpy().ravel()
    values = values[np.isfinite(values)]
    return {lv.value: round(float(np.percentile(values, p)), 2) for lv, p in PERCENTILES.items()}


def climatology(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean daily index per cell for each day of the year, smoothed over 15 days."""
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in daily.index])
    mean = daily.groupby(doy).mean().reindex(range(1, 366))
    wrapped = pd.concat([mean.iloc[-7:], mean, mean.iloc[:7]])
    return wrapped.rolling(15, center=True, min_periods=8).mean().iloc[7:-7]


# --- Forecast -------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _openmeteo(day: date) -> xr.Dataset | None:
    """One clean forecast file, loaded. Kept, because each file serves two IST days."""
    path = store.clean_path("openmeteo", day)
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        return ds[VARIABLES].sel(model=MODELS).load()


@lru_cache(maxsize=1024)
def forecast_daily(day: date) -> pd.DataFrame | None:
    """Each model's daily index at each ward's point (rows wards, columns models)."""
    parts = [_openmeteo(d) for d in (day - timedelta(days=1), day)]
    if any(part is None for part in parts):
        return None
    hourly = xr.concat(parts, dim="time")
    start = np.datetime64(f"{day - timedelta(days=1)}T19:00")  # IST day = 18:30 to 18:30 UTC
    window = hourly.sel(time=slice(start, np.datetime64(f"{day}T18:00")))
    if window.sizes["time"] != 24:
        return None
    at = apparent_temperature(
        window["temperature_2m"], window["relative_humidity_2m"], window["wind_speed_10m"]
    )
    peak = at.max(dim="time", skipna=False)
    points = flood.ward_points()
    lat = xr.DataArray(points["lat"].values, dims="ward")
    lon = xr.DataArray(points["lon"].values, dims="ward")
    near = peak.sel(lat=lat, lon=lon, method="nearest")
    values = peak.interp(lat=lat, lon=lon).fillna(near)  # a coastal cell may be missing
    return pd.DataFrame(
        values.transpose("ward", "model").values, index=points.index, columns=MODELS
    )


def fit_map(forecasts: np.ndarray, truth: np.ndarray) -> dict:
    ok = np.isfinite(forecasts) & np.isfinite(truth)
    return {
        "forecast": np.quantile(forecasts[ok], QUANTILES).round(3).tolist(),
        "era5": np.quantile(truth[ok], QUANTILES).round(3).tolist(),
    }


def apply_map(mapping: dict, values):
    """Replace each forecast value with the ERA5 value at the same rank. Outside the training
    range the end offsets are kept, so a record-hot forecast stays record-hot."""
    fq, oq = np.asarray(mapping["forecast"]), np.asarray(mapping["era5"])
    fq = np.maximum.accumulate(fq + np.arange(fq.size) * 1e-9)
    x = np.asarray(values, dtype=float)
    out = np.interp(x, fq, oq)
    out = np.where(x > fq[-1], x + (oq[-1] - fq[-1]), out)
    out = np.where(x < fq[0], x + (oq[0] - fq[0]), out)
    return np.where(np.isfinite(x), out, np.nan)


def table(period: tuple[date, date], era5: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """One row per ward-day with all three models: each model, their mean, ERA5, baselines."""
    cells = ward_cells()
    rows = []
    day = period[0]
    while day <= period[1]:
        fc = forecast_daily(day)
        yesterday = day - timedelta(days=1)
        if fc is not None and day in era5.index and yesterday in era5.index:
            doy = min(day.timetuple().tm_yday, 365)
            frame = fc.copy()
            frame["cell"] = cells[frame.index].values
            frame["era5"] = era5.loc[day, frame["cell"]].values
            frame["persistence"] = era5.loc[yesterday, frame["cell"]].values
            frame["climatology"] = clim.loc[doy, frame["cell"]].values
            frame["date"] = day
            rows.append(frame.reset_index(names="ward_id"))
        day += timedelta(days=1)
    if not rows:
        raise RuntimeError(f"no forecast days matched ERA5 in {span(period)}")
    out = pd.concat(rows, ignore_index=True)
    out["mean"] = out[MODELS].mean(axis=1)
    return out.dropna(subset=[*MODELS, "era5"])


def span(period: tuple[date, date]) -> str:
    return f"{period[0]:%d %b %Y} to {period[1]:%d %b %Y}"


def fit_maps(rows: pd.DataFrame) -> dict:
    truth = rows["era5"].to_numpy()
    return {m: fit_map(rows[m].to_numpy(), truth) for m in [*MODELS, "mean"]}


def combine(values: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """The candidate forecasts from raw per-model values (columns MODELS)."""
    corrected = np.column_stack([apply_map(maps[m], values[m]) for m in MODELS])
    return pd.DataFrame(
        {
            "corrected_mean": apply_map(maps["mean"], values[MODELS].mean(axis=1)),
            "corrected_median": np.median(corrected, axis=1),
            "corrected_ecmwf": corrected[:, MODELS.index("ecmwf_ifs025")],
        },
        index=values.index,
    )


def correct(rows: pd.DataFrame, maps: dict) -> pd.DataFrame:
    return pd.concat([rows, combine(rows, maps)], axis=1)


# --- Scores ---------------------------------------------------------------------------------


def scores(rows: pd.DataFrame, limits: dict, methods: dict) -> pd.DataFrame:
    """Scored per cell-day: the wards in one ERA5 cell share one truth, so their forecasts are
    averaged first. Counting ward-days would count each case about twelve times."""
    cells = rows.groupby(["date", "cell"])[[*methods, "era5"]].mean()
    out = []
    for key, label in methods.items():
        error = cells[key] - cells["era5"]
        row = {"key": key, "method": label, "mae": error.abs().mean(), "bias": error.mean()}
        for name in ("yellow", "orange"):
            seen, said = cells["era5"] >= limits[name], cells[key] >= limits[name]
            hits = int((seen & said).sum())
            misses = int((seen & ~said).sum())
            false = int((~seen & said).sum())
            row |= {
                f"{name}_hits": hits,
                f"{name}_misses": misses,
                f"{name}_false": false,
                f"{name}_pod": hits / (hits + misses) if hits + misses else np.nan,
                f"{name}_far": false / (hits + false) if hits + false else np.nan,
                f"{name}_csi": hits / (hits + misses + false) if hits + misses + false else np.nan,
            }
        out.append(row)
    return pd.DataFrame(out)


def choose(validation: pd.DataFrame) -> str:
    """Best candidate on the validation year: orange CSI, then yellow CSI, then lower MAE."""
    ranked = validation[validation["key"].isin(CANDIDATES)].sort_values(
        ["orange_csi", "yellow_csi", "mae"], ascending=[False, False, True]
    )
    return str(ranked.iloc[0]["key"])


def days_per_year(daily: pd.DataFrame, limits: dict) -> pd.DataFrame:
    """Days a year on which either cell reaches each level, in five-year blocks."""
    city = daily.max(axis=1)
    years = pd.Series([d.year for d in city.index], index=city.index)
    rows = []
    for start in range(years.min(), years.max() + 1, 5):
        inside = (years >= start) & (years < start + 5)
        block, n = city[inside], years[inside].nunique()
        row = {"years": f"{start}-{min(start + 4, years.max())}"}
        for name in ("yellow", "orange", "red"):
            row[name] = round(float((block >= limits[name]).sum()) / n, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def months_of_heat(daily: pd.DataFrame, limits: dict) -> pd.Series:
    city = daily.max(axis=1)
    hot = city[city >= limits["orange"]]
    return pd.Series([d.month for d in hot.index]).value_counts(normalize=True).sort_index()


# --- Model file and runs --------------------------------------------------------------------


def save_model(limits: dict, maps: dict, method: str, fitted_on: list[str]) -> dict:
    model = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "index": "apparent temperature (Steadman, shade), daily max over the IST day",
        "thresholds": limits,
        "percentiles": {k.value: v for k, v in PERCENTILES.items()},
        "climate_years": list(CLIMATE),
        "method": method,
        "fitted_on": fitted_on,
        "cells": CELLS,
        "maps": maps,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=1), encoding="utf-8")
    return model


@lru_cache(maxsize=1)
def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python -m models.heat --train ...")
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def threshold_text(limits: dict) -> str:
    return (
        f"Apparent temperature, daily max. Yellow from {limits['yellow']:.1f} °C, orange from "
        f"{limits['orange']:.1f}, red from {limits['red']:.1f} (ERA5 1991-2020, percentiles 90, "
        "95, 99). City scale: the inputs are 10-25 km grids, so wards differ little."
    )


def compute(day: date, source: str = "forecast", now: datetime | None = None, live: bool = False):
    """Heat level for every ward for one IST day. Returns the run record and WardRisk rows."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    now = now or datetime.now(UTC)
    model = load_model()
    limits = model["thresholds"]
    if source == "forecast":
        fc = forecast_daily(day)
        if fc is None:
            raise RuntimeError(
                f"forecast missing for {day}. Run first: python -m pipelines.run "
                f"--start {day - timedelta(days=1)} --end {day} --sources openmeteo"
            )
        value = combine(fc, model["maps"])[model["method"]]
        each = np.column_stack([apply_map(model["maps"][m], fc[m]) for m in MODELS])
        probability = pd.Series((each >= limits["orange"]).mean(axis=1), index=fc.index)
        issued = [t for d in (day - timedelta(days=1), day) if (t := risk.forecast_issued(d))]
        if not issued:
            raise RuntimeError(f"no download time recorded for the forecast of {day}")
        data_time = max(issued)
    else:
        era5 = era5_daily(day.year, day.year)
        if day not in era5.index:
            raise RuntimeError(f"no ERA5 for {day}. Run: python -m models.heat --fetch {day.year}")
        value = ward_cells().map(era5.loc[day])
        probability = (value >= limits["orange"]).astype(float)
        data_time = risk.valid_window(day)[1]  # the whole day is observed
    run_id = f"heat-{source}-{day:%Y%m%d}-{now:%Y%m%dT%H%M%SZ}"
    valid_from, valid_to = risk.valid_window(day)
    age = max(0, int((now - data_time).total_seconds() // 60))
    risks = [
        WardRisk(
            ward_id=ward,
            hazard=Hazard.HEAT,
            valid_from=valid_from,
            valid_to=valid_to,
            probability=round(float(probability[ward]), 4),
            severity=level(float(value[ward]), limits),
            trust=TRUST,
            run_id=run_id,
            data_age_minutes=age,
            value=round(float(value[ward]), 1),
            unit=UNIT,
        )
        for ward in value.index
    ]
    run = {
        "run_id": run_id,
        "issued_at": now,
        "data_time": data_time,
        "rain_source": source,
        "model": "heat_at " + model["created_at"],
        "is_test": not (live and source == "forecast"),
        "note": threshold_text(limits)[:500],
    }
    return {"run": run, "risks": risks}


# --- Report ---------------------------------------------------------------------------------


def _row(cells: list) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _num(v, digits: int = 2) -> str:
    return "-" if pd.isna(v) else f"{v:.{digits}f}"


def _score_table(scored: pd.DataFrame) -> list[str]:
    header = [
        "Method",
        "MAE (°C)",
        "Bias (°C)",
        "Yellow POD",
        "Yellow FAR",
        "Yellow CSI",
        "Orange hits",
        "Orange misses",
        "Orange false alarms",
        "Orange CSI",
    ]
    lines = [_row(header), _row(["---"] * len(header))]
    for _, r in scored.iterrows():
        lines.append(
            _row(
                [
                    r["method"],
                    _num(r["mae"]),
                    _num(r["bias"]),
                    _num(r["yellow_pod"]),
                    _num(r["yellow_far"]),
                    _num(r["yellow_csi"]),
                    r["orange_hits"],
                    r["orange_misses"],
                    r["orange_false"],
                    _num(r["orange_csi"]),
                ]
            )
        )
    return lines


def report(limits, validation, scored, rows, per_year, months, hottest, years, method, command):
    fitted, chosen_on, tested = (span(p) for p in years)
    test_year = years[2][0].year if years[2][0].year == years[2][1].year else tested
    best = scored.set_index("key")
    chosen, persistence = best.loc[method], best.loc["persistence"]
    n_cells = rows.groupby(["date", "cell"]).ngroups
    lines = [
        "# Heat stress: apparent temperature per ward",
        "",
        f"Generated by `{command}`. Do not edit by hand.",
        "",
        "## Result",
        "",
        f"- **Levels** (apparent temperature, daily max): yellow from {limits['yellow']:.1f} °C, "
        f"orange from {limits['orange']:.1f} °C, red from {limits['red']:.1f} °C: the 90th, "
        "95th and 99th percentiles of every day in ERA5 1991-2020 over Mumbai. Humid heat keeps "
        "the index high for much of the year, so the levels are less than 2 °C apart and a small "
        "forecast error can move a day across one.",
        f"- **Method:** {CANDIDATES[method]}, chosen on {chosen_on} from the three "
        "candidates below, before the test year was scored.",
        f"- **Day-ahead forecast, {test_year}:** off by {chosen['mae']:.2f} °C on average. It "
        f"caught {chosen['orange_hits']} of {chosen['orange_hits'] + chosen['orange_misses']} "
        f"orange cell-days with {chosen['orange_false']} false alarms (orange CSI "
        f"{_num(chosen['orange_csi'])}). Repeating yesterday's ERA5 value, which is not "
        f"available live, scores CSI {_num(persistence['orange_csi'])}.",
        f"- **Trend:** orange days a year went from {per_year.iloc[0]['orange']} in "
        f"{per_year.iloc[0]['years']} to {per_year.iloc[-1]['orange']} in "
        f"{per_year.iloc[-1]['years']} (table below). Part of a reanalysis trend can come "
        "from changes in the observations it takes in, so treat its size as indicative. With "
        "thresholds fixed on 1991-2020, recent years reach orange far more often than 5% of "
        "days, which the Stage 13 alert rules must weigh against alert fatigue.",
        "- **City scale only.** ERA5 has two land cells over Mumbai and the forecasts are 10-25 km "
        "grids, so neighbouring wards get nearly the same level. Ward-level heat (built-up area, "
        "tree cover) needs land-surface temperature, which is not in this stage.",
        "",
        "## Setup",
        "",
        "- **Index:** apparent temperature (Steadman 1994, shade version used by the Australian "
        "Bureau of Meteorology): AT = T + 0.33 e - 0.70 ws - 4.00, from air temperature, "
        "vapour pressure and 10 m wind. The daily value is the highest hourly value in the IST "
        "day. Direct sun would add several degrees; the index does not include it.",
        "- **ERA5:** hourly reanalysis at 0.25 degree, through Open-Meteo's historical weather "
        "API (no account needed). Over Mumbai the API serves the land cells at 19.00 N and "
        "19.25 N, both at 73.00 E, slightly inland of the city. Each ward takes the nearer cell.",
        "- **Forecast:** ECMWF, GFS and ICON hourly temperature, humidity and wind (Open-Meteo's "
        "archive of short-lead forecasts, so about a day ahead). The index is computed per model "
        "at each ward's point and quantile-mapped to ERA5.",
        f"- **Split by time:** correction fitted on {fitted} and the method chosen on "
        f"{chosen_on}; then the correction refitted on both and scored once on {tested}. The "
        "ECMWF archive starts in February 2024, so there is no earlier training year with all "
        "three models.",
        "- **Probability** on the map is the share of the three corrected models at orange or "
        "above.",
        f"- **Scored per cell-day** ({n_cells} in {test_year}): the wards in one ERA5 cell share "
        "one truth, so their forecasts are averaged first.",
        "- **Baselines:** persistence repeats yesterday's ERA5 value (better than anything "
        "available live, since ERA5 arrives about five days late); climatology is the 1991-2020 "
        "mean for that time of year.",
        "",
        f"## Choosing the method ({chosen_on}, correction fitted on {fitted})",
        "",
        *_score_table(validation),
        "",
        f"## Forecast skill, {tested} (correction fitted on {years[0][0]:%b %Y} to "
        f"{years[1][1]:%b %Y})",
        "",
        *_score_table(scored),
        "",
        "## Days a year at each level (ERA5, either cell)",
        "",
        "Over 1991-2020 each cell is at yellow or above on 10% of days (about 36 a year), orange "
        "on 5% (18) and red on 1% (4). A day counts here when either cell reaches the level, so "
        "the counts run a little higher.",
        "",
        _row(["Years", "Yellow or above", "Orange or above", "Red"]),
        _row(["---"] * 4),
        *[_row([r["years"], r["yellow"], r["orange"], r["red"]]) for _, r in per_year.iterrows()],
        "",
        "## When orange days happen (1991-2020)",
        "",
        _row(["Month", "Share of orange days"]),
        _row(["---"] * 2),
        *[_row([date(2000, m, 1).strftime("%B"), f"{s:.0%}"]) for m, s in months.items()],
        "",
        f"## Hottest days in ERA5, {test_year}",
        "",
        _row(["Date", "Apparent temperature max (°C)", "Level"]),
        _row(["---"] * 3),
        *[_row([d.isoformat(), f"{v:.1f}", level(v, limits).value]) for d, v in hottest.items()],
        "",
        "## Limits",
        "",
        "- The thresholds are statistical, not yet linked to hospital admissions or deaths; the "
        "Stage 13 alert rules should check them against a heat action plan.",
        "- ERA5's nearest land cells sit east of the island city, where afternoons are hotter "
        "and less breezy than on the western seafront.",
        "- Night-time heat (high minimum temperature) is not scored; it matters for health in "
        "humid heat.",
        "- Forecasts are about a day ahead; longer leads will score lower.",
        "- One year of archived forecasts with all three models is little to learn a correction "
        "from: the method chosen on August-December 2024 was not the best one in 2025 (see the "
        "two tables). The correction probably depends on the season.",
        "",
        "## Next",
        "",
        "- Fit the correction by season (pre-monsoon, monsoon, October heat) once 2026 is "
        "archived.",
        "- Add night-time heat (daily minimum apparent temperature) and a wet-bulb check.",
        "- Ward-level heat from Landsat land-surface temperature with the Stage 3 built-up and "
        "green shares (Stage 12, vulnerability).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Heat stress per ward (Stage 9).")
    parser.add_argument("--fetch", nargs=2, type=int, metavar=("FIRST", "LAST"), help="ERA5 years")
    period = {"nargs": 2, "type": date.fromisoformat, "metavar": ("START", "END")}
    parser.add_argument("--train", **period, help="period to fit the forecast correction on")
    parser.add_argument("--validate", **period, help="later period to choose the method on")
    parser.add_argument("--test", **period, help="later period to score on, once")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--date", help="IST date for a heat run, YYYY-MM-DD or 'today'")
    parser.add_argument("--source", choices=SOURCES, default="forecast")
    parser.add_argument("--live", action="store_true", help="do not mark a forecast run TEST")
    args = parser.parse_args(argv)
    if not (args.fetch or args.train or args.date):
        parser.error("give --fetch, --train or --date")

    if args.fetch:
        for year in range(args.fetch[0], args.fetch[1] + 1):
            had = _year_file(year)
            path = fetch_year(year)
            print(f"{year}  {path.relative_to(settings.raw_dir).as_posix()}", flush=True)
            if path != had and year < args.fetch[1]:
                time.sleep(2)  # spread requests: the free API limits calls per minute

    if args.train:
        periods = (tuple(args.train), tuple(args.validate or ()), tuple(args.test or ()))
        if not (args.validate and args.test) or not (
            periods[0][1] < periods[1][0] and periods[1][1] < periods[2][0]
        ):
            parser.error("need --train, then a later --validate, then a later --test")
        climate = era5_daily(*CLIMATE)
        limits = thresholds(climate)
        recent = era5_daily(periods[0][0].year, periods[2][1].year)
        clim = climatology(climate)
        first = table(periods[0], recent, clim)
        middle = table(periods[1], recent, clim)
        validation = scores(correct(middle, fit_maps(first)), limits, CANDIDATES)
        method = choose(validation)
        maps = fit_maps(pd.concat([first, middle], ignore_index=True))
        save_model(limits, maps, method, [periods[0][0].isoformat(), periods[1][1].isoformat()])
        load_model.cache_clear()
        test_rows = correct(table(periods[2], recent, clim), maps)
        scored = scores(test_rows, limits, {**CANDIDATES, **REFERENCE})
        print(f"Thresholds: {limits}")
        print(validation.drop(columns="key").round(2).to_string(index=False))
        print(f"\nChosen on {span(periods[1])}: {CANDIDATES[method]}\n")
        print(scored.drop(columns="key").round(2).to_string(index=False))
        print(f"Saved {MODEL_PATH}")
        if args.report:
            everything = era5_daily(CLIMATE[0], datetime.now(UTC).year - 1)
            test_year = recent[[periods[2][0] <= d <= periods[2][1] for d in recent.index]]
            hottest = test_year.max(axis=1).sort_values(ascending=False).head(8)
            command = "python -m models.heat " + " ".join(
                argv if argv is not None else sys.argv[1:]
            )
            text = report(
                limits,
                validation,
                scored,
                test_rows,
                days_per_year(everything, limits),
                months_of_heat(climate, limits),
                hottest,
                periods,
                method,
                command,
            )
            args.report.write_text(text, encoding="utf-8")
            print(f"Wrote {args.report}")

    if args.date:
        day = (
            datetime.now(risk.IST).date() if args.date == "today" else date.fromisoformat(args.date)
        )
        try:
            result = compute(day, args.source, live=args.live)
        except (RuntimeError, FileNotFoundError) as exc:
            print(exc)
            return 1
        run_id = risk.save(result)
        values = [r.value for r in result["risks"]]
        counts = pd.Series([r.severity.value for r in result["risks"]]).value_counts().to_dict()
        print(f"Saved run {run_id} ({'TEST' if result['run']['is_test'] else 'LIVE'})")
        print(f"Apparent temperature max {min(values):.1f} to {max(values):.1f} °C; {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
