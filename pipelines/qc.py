"""Schema, range and gap checks on harmonised data, run before anything reaches the clean zone.

Schema  the variables, dimensions, units, grid and time steps each source must have.
        A schema failure is an error: the file is not written.
Range   physical limits for the pilot area. Values outside them are set to missing and counted.
Gap     missing time steps, missing values, and (for a date range) missing days.

python -m pipelines.qc --start 2025-07-19 --end 2025-07-21     re-check clean files already on disk
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import xarray as xr

from api.config import settings
from pipelines import store
from pipelines.grid import target_coords

QC_LOG = "qc_log.jsonl"
MISSING_WARN = 0.2  # warn when more than this fraction of a variable is missing


@dataclass(frozen=True)
class VarSpec:
    units: str
    low: float
    high: float


@dataclass(frozen=True)
class SourceSpec:
    dims: tuple[str, ...]
    steps: int  # time steps per day
    on_grid: bool
    variables: dict[str, VarSpec]


# Limits are wide on purpose: they catch impossible values, not unusual weather.
# Mumbai records for scale: 944 mm in a day (26 July 2005), about 190 mm in an hour,
# air temperature between about 7 and 43 degC.
SPECS: dict[str, SourceSpec] = {
    "openmeteo": SourceSpec(
        dims=("model", "time", "lat", "lon"),
        steps=24,
        on_grid=True,
        variables={
            "precipitation": VarSpec("mm", 0, 250),
            "temperature_2m": VarSpec("degC", 0, 50),
            "relative_humidity_2m": VarSpec("%", 0, 100),
            "wind_speed_10m": VarSpec("m s-1", 0, 75),
        },
    ),
    "tide": SourceSpec(
        dims=("time",),
        steps=24,
        on_grid=False,
        variables={"sea_level": VarSpec("m", -4, 4)},
    ),
    "chirps": SourceSpec(
        dims=("time", "lat", "lon"),
        steps=1,
        on_grid=True,
        variables={"precipitation": VarSpec("mm", 0, 1200)},
    ),
    "imd": SourceSpec(
        dims=("time", "lat", "lon"),
        steps=1,
        on_grid=True,
        variables={"precipitation": VarSpec("mm", 0, 1200)},
    ),
}


@dataclass(frozen=True)
class Issue:
    check: str  # "schema", "range" or "gap"
    level: str  # "error" or "warning"
    variable: str  # "" when the issue is about the whole file
    detail: str
    count: int = 0

    def __str__(self) -> str:
        where = f" {self.variable}:" if self.variable else ""
        return f"{self.check} {self.level}:{where} {self.detail}"


class SchemaError(ValueError):
    """The dataset does not have the agreed shape, so it must not enter the clean zone."""


def check_schema(source: str, ds: xr.Dataset, day: date) -> list[Issue]:
    spec = SPECS[source]
    issues: list[Issue] = []

    def bad(detail: str, variable: str = "") -> None:
        issues.append(Issue("schema", "error", variable, detail))

    for name, var in spec.variables.items():
        if name not in ds.data_vars:
            bad("variable is missing", name)
            continue
        if ds[name].dims != spec.dims:
            bad(f"dimensions are {ds[name].dims}, expected {spec.dims}", name)
        if ds[name].attrs.get("units") != var.units:
            bad(f"units are {ds[name].attrs.get('units')!r}, expected {var.units!r}", name)
        if not np.issubdtype(ds[name].dtype, np.floating):
            bad(f"type is {ds[name].dtype}, expected a float", name)
    for name in sorted(set(ds.data_vars) - set(spec.variables)):
        bad("variable is not in the schema", name)

    if spec.on_grid:
        lons, lats = target_coords()
        for coord, want in (("lon", lons), ("lat", lats)):
            have = ds[coord].values if coord in ds.coords else np.array([])
            if have.shape != want.shape or not np.allclose(have, want, atol=1e-6):
                bad(f"{coord} does not match the pilot grid")

    if "time" not in ds.coords:
        bad("time coordinate is missing")
        return issues
    times = ds["time"].values
    if not np.issubdtype(times.dtype, np.datetime64):
        bad(f"time type is {times.dtype}, expected datetime64")
        return issues
    if times.size > 1 and not (np.diff(times) > np.timedelta64(0)).all():
        bad("time is not strictly increasing")
    start = np.datetime64(day.isoformat(), "ns")
    outside = int(((times < start) | (times >= start + np.timedelta64(1, "D"))).sum())
    if outside:
        bad(f"{outside} time steps fall outside {day} (UTC)")
    return issues


def check_range(source: str, ds: xr.Dataset) -> list[Issue]:
    issues = []
    for name, var in SPECS[source].variables.items():
        if name not in ds.data_vars:
            continue
        values = ds[name].values
        count = int(((values < var.low) | (values > var.high)).sum())
        if count:
            worst = float(np.nanmax(np.abs(values)))
            detail = f"{count} values outside [{var.low:g}, {var.high:g}] {var.units}"
            issues.append(Issue("range", "warning", name, f"{detail} (largest {worst:g})", count))
    return issues


def mask_out_of_range(source: str, ds: xr.Dataset) -> xr.Dataset:
    """Set out-of-range values to missing. Returns a new dataset; the input is not changed."""
    ds = ds.copy()
    for name, var in SPECS[source].variables.items():
        if name in ds.data_vars:
            inside = (ds[name] >= var.low) & (ds[name] <= var.high)
            ds[name] = ds[name].where(inside).assign_attrs(ds[name].attrs)
    return ds


def check_gaps(source: str, ds: xr.Dataset, day: date) -> list[Issue]:
    spec = SPECS[source]
    issues = []
    if "time" in ds.coords:
        step = np.timedelta64(24 * 3600 // spec.steps, "s")
        expected = np.datetime64(day.isoformat(), "ns") + step * np.arange(spec.steps)
        missing = int((~np.isin(expected, ds["time"].values)).sum())
        if missing:
            detail = f"{missing} of {spec.steps} time steps are missing"
            issues.append(Issue("gap", "warning", "", detail, missing))
    for name in spec.variables:
        if name not in ds.data_vars or ds[name].size == 0:
            continue
        nan = ds[name].isnull()
        frac = float(nan.mean())
        if frac == 1:
            issues.append(Issue("gap", "warning", name, "every value is missing", int(nan.sum())))
        elif frac > MISSING_WARN:
            detail = f"{frac:.0%} of values are missing"
            issues.append(Issue("gap", "warning", name, detail, int(nan.sum())))
    return issues


def validate(source: str, ds: xr.Dataset, day: date) -> tuple[xr.Dataset, list[Issue]]:
    """Run all checks. Returns the dataset with out-of-range values masked, and the issues found.

    Raises SchemaError when the schema is wrong.
    """
    schema = check_schema(source, ds, day)
    if schema:
        raise SchemaError("; ".join(str(i) for i in schema))
    range_issues = check_range(source, ds)
    cleaned = mask_out_of_range(source, ds) if range_issues else ds
    issues = range_issues + check_gaps(source, cleaned, day)
    cleaned.attrs["qc_masked_values"] = sum(i.count for i in range_issues)
    return cleaned, issues


def log_issues(source: str, day: date, issues: list[Issue]) -> None:
    """Append one line per issue to processed/qc_log.jsonl."""
    if not issues:
        return
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with (settings.processed_dir / QC_LOG).open("a", encoding="utf-8") as f:
        for issue in issues:
            entry = {"source": source, "day": day.isoformat(), "checked_at": stamp, **asdict(issue)}
            f.write(json.dumps(entry) + "\n")


def check_clean_file(source: str, day: date) -> list[Issue] | None:
    """Re-check a clean file on disk. None when the file does not exist."""
    path = store.clean_path(source, day)
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        ds.load()
    schema = check_schema(source, ds, day)
    return schema or check_range(source, ds) + check_gaps(source, ds, day)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-check clean files for a date range.")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--sources", default=",".join(SPECS))
    args = parser.parse_args(argv)
    end = args.end or args.start
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = sorted(set(sources) - set(SPECS))
    if unknown:
        parser.error(f"unknown sources {unknown}; choose from {sorted(SPECS)}")

    days = [args.start + timedelta(days=n) for n in range((end - args.start).days + 1)]
    errors = 0
    print(f"{'source':<10} {'days':>5} {'missing':>8} {'schema':>7} {'range':>6} {'gap':>4}")
    for source in sources:
        counts = {"schema": 0, "range": 0, "gap": 0}
        missing_days = []
        for day in days:
            issues = check_clean_file(source, day)
            if issues is None:
                missing_days.append(day)
                continue
            for issue in issues:
                counts[issue.check] += 1
                errors += issue.level == "error"
                print(f"  {day}  {source}: {issue}")
        found = len(days) - len(missing_days)
        print(
            f"{source:<10} {found:>5} {len(missing_days):>8} "
            f"{counts['schema']:>7} {counts['range']:>6} {counts['gap']:>4}"
        )
        if missing_days:
            shown = ", ".join(d.isoformat() for d in missing_days[:5])
            more = f" and {len(missing_days) - 5} more" if len(missing_days) > 5 else ""
            print(f"  missing days: {shown}{more}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
