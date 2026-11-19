"""Flood model: chance that a ward floods on a day, from rain, tide and ward features (XGBoost).

One row is a ward-day from labels/flood_events.csv. Features:
  rain_overnight_mm   IMD rain in the 24 h ending 08:30 IST on the day (the night before)
  rain_today_mm       IMD rain in the 24 h ending 08:30 IST the next day (the day itself)
  rain_prior3_mm      IMD rain in the three IMD days before rain_overnight (how wet the ground was)
  tide_high_m         highest astronomical tide in the day, from models/tide.py
  ward features       from Stage 3 (data/processed/static/ward_features.csv)
Rain is read at one point inside each ward from the clean IMD grid. IMD is 0.25 degree, so
neighbouring wards get almost the same rain: ward differences come from the ward features.

The model is trained on observed rain. In live use the rain comes from the Stage 4 corrected
forecast, which is less accurate, so live skill will be lower than the scores here.

Split by time, never randomly:
  held-out years    train on years before --test-from, score the years from it on
  year by year      for each labelled year, train on all earlier years and score that year
Baseline: "flood if IMD shows heavy rain (64.5 mm or more) in either 24 h window".

python -m models.flood --test-from 2024 --report docs/flood_model.md
python -m models.flood --predict 2025-08-19
"""

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBClassifier

from api.config import settings
from models import tide
from pipelines import labels, store

STATIC_PATH = settings.processed_dir / "static" / "ward_features.csv"
MODEL_PATH = settings.processed_dir / "models" / "flood_xgb.json"
META_PATH = settings.processed_dir / "models" / "flood_xgb_meta.json"

RAIN = ["rain_overnight_mm", "rain_today_mm", "rain_prior3_mm"]
TIDE = ["tide_high_m"]
STATIC = [
    "area_km2",
    "elevation_mean_m",
    "elevation_p10_m",
    "low_lying_frac",
    "slope_mean_deg",
    "drain_distance_mean_m",
    "drain_km_per_km2",
    "built_up_frac",
    "green_frac",
    "water_frac",
    "wetland_frac",
]
FEATURE_SETS = {"rain only": RAIN, "rain + tide": RAIN + TIDE, "all": RAIN + TIDE + STATIC}
# +1: more of this can only raise the flood chance; -1: can only lower it; 0: free.
# With so few labels this stops the model learning physically backwards rules.
DIRECTION = {
    "rain_overnight_mm": 1,
    "rain_today_mm": 1,
    "rain_prior3_mm": 1,
    "tide_high_m": 1,
    "low_lying_frac": 1,
    "elevation_mean_m": -1,
    "elevation_p10_m": -1,
}
HEAVY_MM = 64.5  # IMD "heavy rain" in 24 h
THRESHOLD = 0.5  # a ward-day is flagged when the probability reaches this
SEASON = (6, 9)  # June to September, for counting how often the model would warn


# --- Features ------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def ward_points() -> pd.DataFrame:
    """One point inside each ward (lat, lon), indexed by ward_id."""
    wards = labels.load_wards()
    if wards is None:
        raise FileNotFoundError("Ward file not found. Run: python -m scripts.load_wards")
    points = wards.to_crs(32643).representative_point().to_crs(4326)
    return pd.DataFrame({"lat": points.y.values, "lon": points.x.values}, index=wards["ward_id"])


@lru_cache(maxsize=4096)
def imd_at_wards(day: date) -> pd.Series | None:
    """IMD rain (mm) for the 24 h ending 08:30 IST on `day`, at each ward's point."""
    path = store.clean_path("imd", day)
    if not path.exists():
        return None
    points = ward_points()
    with xr.open_dataset(path) as ds:
        field = ds["precipitation"].isel(time=0).load()
    lat = xr.DataArray(points["lat"].values, dims="ward")
    lon = xr.DataArray(points["lon"].values, dims="ward")
    values = field.interp(lat=lat, lon=lon).values
    nearest = field.sel(lat=lat, lon=lon, method="nearest").values
    values = np.where(np.isnan(values), nearest, values)  # a coastal cell may be missing
    return pd.Series(values, index=points.index, dtype=float)


def rain_features(day: date, ward_id: str) -> dict | None:
    """The three rain features for one ward-day, or None if an IMD day is missing."""
    needed = {offset: imd_at_wards(day + timedelta(days=offset)) for offset in (-3, -2, -1, 0, 1)}
    if any(v is None for v in needed.values()):
        return None
    return {
        "rain_overnight_mm": needed[0][ward_id],
        "rain_today_mm": needed[1][ward_id],
        "rain_prior3_mm": needed[-3][ward_id] + needed[-2][ward_id] + needed[-1][ward_id],
    }


def static_features() -> pd.DataFrame:
    if not STATIC_PATH.exists():
        raise FileNotFoundError(f"{STATIC_PATH} not found. Run: python -m pipelines.static")
    return pd.read_csv(STATIC_PATH).set_index("ward_id")[STATIC]


def build(ward_days: pd.DataFrame, tide_model: dict | None = None) -> pd.DataFrame:
    """Add feature columns to a table with `date` and `ward_id`. Rows without rain are dropped
    and listed in `.attrs["dropped"]`."""
    tide_model = tide_model or tide.load_model()
    rows, dropped = [], []
    for record in ward_days.to_dict("records"):
        rain = rain_features(record["date"], record["ward_id"])
        if rain is None:
            dropped.append((record["date"], record["ward_id"]))
            continue
        rows.append({**record, **rain})
    table = pd.DataFrame(rows, columns=list(ward_days.columns) + RAIN)
    highs = tide.daily_high(tide_model, sorted(set(table["date"])))
    table["tide_high_m"] = table["date"].map(highs)
    table = table.join(static_features(), on="ward_id")
    table.attrs["dropped"] = dropped
    return table


def labelled_table() -> pd.DataFrame:
    table = build(labels.ward_days(labels.read()))
    table["year"] = [d.year for d in table["date"]]
    return table


def season_table(year: int) -> pd.DataFrame:
    """Every ward on every day of the monsoon season (June to September) of one year."""
    start, end = date(year, SEASON[0], 1), date(year, SEASON[1] + 1, 1) - timedelta(days=1)
    days = [start + timedelta(days=n) for n in range((end - start).days + 1)]
    grid = pd.DataFrame(
        [(d, w) for d in days for w in ward_points().index], columns=["date", "ward_id"]
    )
    return build(grid)


# --- Model and scores ----------------------------------------------------------------------


def make_model(features: list[str]) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=1,
        reg_lambda=1.0,
        monotone_constraints=tuple(DIRECTION.get(f, 0) for f in features),
        eval_metric="logloss",
        random_state=0,
        n_jobs=1,
    )


def fit(train: pd.DataFrame, features: list[str]) -> XGBClassifier:
    if train["flooded"].nunique() < 2:
        raise ValueError("training data needs both flooded and not-flooded ward-days")
    return make_model(features).fit(train[features], train["flooded"])


def rule(table: pd.DataFrame) -> np.ndarray:
    """Baseline: flag when either 24 h IMD window has heavy rain."""
    heavy = table[["rain_overnight_mm", "rain_today_mm"]].max(axis=1) >= HEAVY_MM
    return heavy.to_numpy(dtype=float)


def scores(flooded, flagged, probability=None) -> dict:
    """Counts and scores for yes/no warnings.

    POD  = share of floods that were flagged
    FAR  = false alarm ratio: share of flags where no flood was reported
    POFD = false alarm rate: share of not-flooded ward-days that were flagged
    CSI  = hits / (hits + misses + false alarms)
    """
    y = np.asarray(flooded).astype(bool)
    f = np.asarray(flagged).astype(bool)
    hits, misses = int((y & f).sum()), int((y & ~f).sum())
    false_alarms, correct_no = int((~y & f).sum()), int((~y & ~f).sum())

    def ratio(a, b):
        return a / b if b else float("nan")

    out = {
        "n": len(y),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_no": correct_no,
        "pod": ratio(hits, hits + misses),
        "far": ratio(false_alarms, hits + false_alarms),
        "pofd": ratio(false_alarms, false_alarms + correct_no),
        "csi": ratio(hits, hits + misses + false_alarms),
    }
    if probability is not None:
        out["brier"] = float(np.mean((np.asarray(probability) - y) ** 2))
    return out


def held_out(table: pd.DataFrame, test_from: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on years before test_from, score the rest. Returns (scores, per-row predictions)."""
    train, test = table[table["year"] < test_from], table[table["year"] >= test_from]
    results, rows = [], test[["date", "ward_id", "flooded"] + RAIN].copy()
    for name, features in FEATURE_SETS.items():
        p = fit(train, features).predict_proba(test[features])[:, 1]
        rows[f"p_{name}"] = p
        results.append({"method": f"XGBoost, {name}", **scores(test["flooded"], p >= THRESHOLD, p)})
    rows["rule"] = rule(test)
    results.append({"method": "heavy-rain rule", **scores(test["flooded"], rule(test))})
    return pd.DataFrame(results), rows


def year_by_year(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """For each year, train only on earlier years. Years with no usable history are skipped."""
    out = []
    for year in sorted(table["year"].unique()):
        train, test = table[table["year"] < year], table[table["year"] == year]
        if train["flooded"].nunique() < 2:
            continue
        p = fit(train, features).predict_proba(test[features])[:, 1]
        model = scores(test["flooded"], p >= THRESHOLD)
        base = scores(test["flooded"], rule(test))
        out.append(
            {
                "year": year,
                "trained_on": f"{int(train['year'].min())}-{int(train['year'].max())}",
                "flooded": int(test["flooded"].sum()),
                "not_flooded": int((test["flooded"] == 0).sum()),
                "model_hits": model["hits"],
                "model_false_alarms": model["false_alarms"],
                "rule_hits": base["hits"],
                "rule_false_alarms": base["false_alarms"],
            }
        )
    return pd.DataFrame(out)


def warning_counts(model: XGBClassifier, features: list[str], year: int, table: pd.DataFrame):
    """How often the model and the rule would warn over a whole season, against the labels."""
    season = season_table(year)
    if season.empty:
        return None
    p = model.predict_proba(season[features])[:, 1]
    flags = {"model": p >= THRESHOLD, "rule": rule(season).astype(bool)}
    known = table[(table["year"] == year) & (table["flooded"] == 1)]
    known = set(zip(known["date"], known["ward_id"], strict=True))
    in_season = {k for k in known if SEASON[0] <= k[0].month <= SEASON[1]}
    keys = list(zip(season["date"], season["ward_id"], strict=True))
    out = {
        "year": year,
        "ward_days": len(season),
        "days": season["date"].nunique(),
        "reported": len(in_season),
    }
    for name, f in flags.items():
        flagged = {k for k, hit in zip(keys, f, strict=True) if hit}
        out[f"{name}_ward_days"] = len(flagged)
        out[f"{name}_days"] = len({d for d, _ in flagged})
        out[f"{name}_caught"] = len(flagged & in_season)
    return out


# --- Final model, prediction and report ---------------------------------------------------


def train_final(table: pd.DataFrame, features: list[str]) -> XGBClassifier:
    model = fit(table, features)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    meta = {
        "features": features,
        "threshold": THRESHOLD,
        "trained_on_years": sorted(int(y) for y in table["year"].unique()),
        "ward_days": int(len(table)),
        "flooded": int(table["flooded"].sum()),
        "rain_source": "IMD gridded daily, observed",
        "tide_model": str(tide.MODEL_PATH.relative_to(settings.data_dir.parent).as_posix()),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "labels_reviewed": int((labels.read()["reviewed"] == "yes").sum()),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model


def load_final() -> tuple[XGBClassifier, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python -m models.flood")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model, json.loads(META_PATH.read_text(encoding="utf-8"))


def predict_day(day: date) -> pd.DataFrame:
    """Flood probability for every ward on a past day, from observed IMD rain (hindcast)."""
    model, meta = load_final()
    grid = pd.DataFrame([(day, w) for w in ward_points().index], columns=["date", "ward_id"])
    table = build(grid)
    if table.empty:
        raise RuntimeError(f"IMD rain missing around {day}; run pipelines.run with --sources imd")
    table["probability"] = model.predict_proba(table[meta["features"]])[:, 1]
    table["flagged"] = table["probability"] >= meta["threshold"]
    return table[["ward_id", *RAIN, "tide_high_m", "probability", "flagged"]]


def _fmt(value, digits=2) -> str:
    if isinstance(value, float):
        return "n/a" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for record in frame.to_dict("records"):
        lines.append("| " + " | ".join(_fmt(record[c]) for c in columns) + " |")
    return lines


def splits(model: XGBClassifier) -> dict[str, list[float]]:
    """The cut points each feature is split at, across all trees."""
    trees = model.get_booster().trees_to_dataframe()
    trees = trees[trees["Feature"] != "Leaf"]
    return {f: sorted(set(g["Split"].round(1))) for f, g in trees.groupby("Feature")}


def verdict(counts: list[dict], cuts: dict[str, list[float]]) -> list[str]:
    lines = []
    if counts:
        seasons = ", ".join(
            f"{c['year']}: model {c['model_days']} days, rule {c['rule_days']} days "
            f"(of {c['days']})"
            for c in counts
        )
        lines.append(
            "- **Season false alarms.** Days on which the model flags at least one ward, against "
            f"the heavy-rain rule: {seasons}. Both flag every reported flood."
        )
        model_days = sum(c["model_days"] for c in counts)
        rule_days = sum(c["rule_days"] for c in counts)
        if model_days > rule_days:
            lines.append(
                f"- **The model warns on {model_days / rule_days:.1f} times as many days as the "
                "rule, so it is not yet better than the rule.** Use the rule if a warning is "
                "needed before the labels improve."
            )
    if len(cuts) == 1:
        feature, values = next(iter(cuts.items()))
        lines.append(
            f"- **The final model uses one feature, `{feature}`, cut at "
            f"{', '.join(f'{v:g}' for v in values)}.** With only dry days as negatives, the "
            'labels teach nothing beyond "some rain", and tide and ward features add nothing.'
        )
    return lines


def report(
    table, test_from, results, predictions, by_year, counts, importance, cuts, command
) -> str:
    train = table[table["year"] < test_from]
    test = table[table["year"] >= test_from]
    reviewed = int((labels.read()["reviewed"] == "yes").sum())
    tide_check = tide.load_model().get("check", {})
    lines = [
        "# Flood model: XGBoost on rain, tide and ward features",
        "",
        f"Generated by `{command}`. Do not edit by hand.",
        "",
        "## Result",
        "",
        *verdict(counts, cuts),
        "",
        "## Read this first",
        "",
        f"- **The labels are few and unchecked.** {len(table)} ward-days "
        f"({int(table['flooded'].sum())} flooded) on {table['date'].nunique()} dates. "
        f"{reviewed} label rows have been reviewed by a person. Every score below moves a lot if "
        "a handful of rows change.",
        "- **The not-flooded ward-days are all dry days** (under 2.5 mm for three days). Any "
        "method that looks at rain gets them right, so the false-alarm scores in the first table "
        "say almost nothing. The season counts further down are the better guide to false alarms.",
        "- **Rain is observed, not forecast.** Live use will feed the model forecast rain, which "
        "is less accurate (see `docs/forecast_skill.md`).",
        "- **Decision support only.** Official warnings remain with IMD, NDMA and the state.",
        "",
        "## Setup",
        "",
        "- **Features:** IMD rain in the 24 h before 08:30 IST on the day and in the 24 h after "
        "it, IMD rain over the three days before that, the day's highest astronomical tide, and "
        "the Stage 3 ward features. IMD is a 0.25 degree grid, so neighbouring wards get nearly "
        "the same rain.",
        "- **Tide:** harmonic fit to Open-Meteo sea level for "
        f"{' to '.join(tide.load_model().get('fitted_on', []))}; on "
        f"{' to '.join(tide_check.get('period', []))}, which it was not fitted on, the daily "
        f"high tide is off by {_fmt(tide_check.get('daily_high_mae_m', float('nan')))} m on "
        "average. Storm surge is not included.",
        "- **Model:** XGBoost, 150 trees of depth 2, with rain, tide, low-lying share and "
        f"elevation forced to push the flood chance in the physical direction only. A ward-day "
        f"is flagged when the probability is {THRESHOLD} or more.",
        "- **Baseline:** flag a ward-day when either 24 h IMD window has heavy rain "
        f"({HEAVY_MM} mm or more).",
        "",
        "## Held-out years",
        "",
        f"Trained on {int(train['year'].min())}-{int(train['year'].max())} "
        f"({len(train)} ward-days, {int(train['flooded'].sum())} flooded); scored on "
        f"{int(test['year'].min())}-{int(test['year'].max())} ({len(test)} ward-days, "
        f"{int(test['flooded'].sum())} flooded, {int((test['flooded'] == 0).sum())} not flooded).",
        "",
        "POD = share of reported floods that were flagged. FAR (false alarm ratio) = share of "
        "flags with no flood. POFD (false alarm rate) = share of not-flooded ward-days that were "
        "flagged. Brier = mean squared error of the probability (lower is better).",
        "",
    ]
    lines += _table(
        results,
        ["method", "hits", "misses", "false_alarms", "correct_no", "pod", "far", "pofd", "csi"],
        ["Method", "Hits", "Misses", "False alarms", "Correct no", "POD", "FAR", "POFD", "CSI"],
    )
    brier = results.dropna(subset=["brier"]) if "brier" in results else results.iloc[0:0]
    if len(brier):
        lines += [
            "",
            "Brier score: "
            + "; ".join(f"{r['method']} {r['brier']:.3f}" for r in brier.to_dict("records"))
            + ".",
        ]
    missed = predictions[(predictions["flooded"] == 1) & (predictions["p_all"] < THRESHOLD)]
    if len(missed):
        lines += ["", "Reported floods the full model missed:", ""]
        lines += _table(
            missed.assign(date=missed["date"].astype(str)),
            ["date", "ward_id", "rain_overnight_mm", "rain_today_mm", "p_all"],
            ["Date", "Ward", "Rain before 08:30 (mm)", "Rain after 08:30 (mm)", "Probability"],
        )
    lines += [
        "",
        "## Year by year",
        "",
        "Each year scored by the full model trained only on the years before it.",
        "",
    ]
    lines += _table(
        by_year,
        [
            "year",
            "trained_on",
            "flooded",
            "not_flooded",
            "model_hits",
            "model_false_alarms",
            "rule_hits",
            "rule_false_alarms",
        ],
        [
            "Year",
            "Trained on",
            "Flooded",
            "Not flooded",
            "Model hits",
            "Model false alarms",
            "Rule hits",
            "Rule false alarms",
        ],
    )
    lines += [
        "",
        "## How often it would warn in a whole season",
        "",
        f"The full model trained on years before {test_from}, run on every ward and every day of "
        "June to September. Only a few ward-days in a season have a flood report, and a ward "
        "with no report is unknown, not dry. So every flag outside a reported ward-day is "
        "either a false alarm or an unreported flood: the count is an upper limit on false "
        "alarms.",
        "",
    ]
    if counts:
        lines += _table(
            pd.DataFrame(counts),
            [
                "year",
                "ward_days",
                "reported",
                "model_ward_days",
                "model_days",
                "model_caught",
                "rule_ward_days",
                "rule_days",
                "rule_caught",
            ],
            [
                "Year",
                "Ward-days",
                "Reported floods",
                "Model flags",
                "Days with a model flag",
                "Reports flagged by model",
                "Rule flags",
                "Days with a rule flag",
                "Reports flagged by rule",
            ],
        )
    else:
        lines.append("No season had complete IMD data.")
    lines += [
        "",
        "## What the final model leans on",
        "",
        "The saved model (`data/processed/models/flood_xgb.json`) is trained on every labelled "
        "year. Share of total gain per feature:",
        "",
    ]
    lines += _table(importance, ["feature", "share"], ["Feature", "Share of gain"])
    lines += [
        "",
        "## Next",
        "",
        "- Review the label rows and add BMC waterlogging lists: heavy-rain days on which a ward "
        "stayed dry are what the model most needs.",
        "- Score the model on the Stage 4 corrected forecast instead of observed rain.",
        "- Add rain at a finer scale than IMD (IMERG, GSMaP) once those accounts exist.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and score the flood model.")
    parser.add_argument("--test-from", type=int, default=2024, help="first held-out year")
    parser.add_argument("--report", type=Path, help="write a Markdown report here")
    parser.add_argument("--predict", type=date.fromisoformat, help="hindcast every ward on a day")
    args = parser.parse_args(argv)

    if args.predict:
        print(predict_day(args.predict).round(3).to_string(index=False))
        return 0

    table = labelled_table()
    for day, ward in table.attrs["dropped"]:
        print(f"skipped {day} {ward}: IMD rain missing (run pipelines.run --sources imd)")
    train = table[table["year"] < args.test_from]
    if train["flooded"].nunique() < 2 or table["year"].max() < args.test_from:
        parser.error("need both classes before --test-from and labels from it on")

    results, predictions = held_out(table, args.test_from)
    features = FEATURE_SETS["all"]
    by_year = year_by_year(table, features)
    trained = fit(train, features)
    counts = []
    for year in sorted(table.loc[table["year"] >= args.test_from, "year"].unique()):
        found = warning_counts(trained, features, int(year), table)
        if found:
            counts.append(found)
    final = train_final(table, features)
    gain = final.get_booster().get_score(importance_type="total_gain")
    importance = pd.DataFrame({"feature": features, "gain": [gain.get(f, 0.0) for f in features]})
    importance["share"] = importance["gain"] / importance["gain"].sum()
    importance = importance.sort_values("share", ascending=False)

    pd.set_option("display.width", 160)
    cols = ["method", "hits", "misses", "false_alarms", "correct_no", "pod", "far", "pofd", "csi"]
    print(results[cols].round(2).to_string(index=False))
    print()
    print(by_year.to_string(index=False))
    print()
    if counts:
        print(pd.DataFrame(counts).to_string(index=False))
    print(f"\nSaved {MODEL_PATH}")
    if args.report:
        command = "python -m models.flood " + " ".join(argv if argv is not None else sys.argv[1:])
        text = report(
            table,
            args.test_from,
            results,
            predictions,
            by_year,
            counts,
            importance,
            splits(final),
            command,
        )
        args.report.write_text(text, encoding="utf-8")
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
