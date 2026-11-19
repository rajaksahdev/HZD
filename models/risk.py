"""Risk run: flood risk for every ward for one local day, written to PostGIS for the API and map.

Rain modes:
  observed   IMD rain for every window: a hindcast. Always marked TEST.
  forecast   Stage 4 corrected forecast (quantile-mapped mean of ECMWF, GFS and ICON) for the day
             itself; earlier days use IMD where it exists, else the forecast.

The probability comes from the Stage 5 model (data/processed/models/flood_xgb.json). Severity
bands are provisional until the Stage 13 alert rules. Trust is low while no flood label has
been reviewed and Stage 5 shows the model warning far more often than a heavy-rain rule.

python -m models.risk --date 2025-08-19 --rain observed
python -m models.risk --date 2026-11-19 --rain forecast
python -m models.risk --list                       # runs in the database
python -m models.risk --delete RUN_ID
"""

import argparse
import json
import re
import sys
from datetime import UTC, date, datetime, time, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import delete, select

from api.db import RiskRun, SessionLocal, WardRiskRow, init_db
from api.schemas import Hazard, Severity, Trust, WardRisk
from models import flood, forecast
from pipelines import store
from pipelines.grid import target_coords

IST = timezone(timedelta(hours=5, minutes=30))
IMD_DAY_END = time(3, 0, tzinfo=UTC)  # an IMD rain day ends 08:30 IST = 03:00 UTC
# Probability at or above each value gives that severity. Provisional until Stage 13.
SEVERITY_BANDS = [(0.75, Severity.RED), (0.5, Severity.ORANGE), (0.25, Severity.YELLOW)]
MODES = ("observed", "forecast")


def severity(probability: float) -> Severity:
    for floor, level in SEVERITY_BANDS:
        if probability >= floor:
            return level
    return Severity.GREEN


def valid_window(day: date) -> tuple[datetime, datetime]:
    """The local calendar day in UTC: 00:00 IST to 00:00 IST the next day."""
    start = datetime.combine(day, time(0, 0), tzinfo=IST).astimezone(UTC)
    return start, start + timedelta(days=1)


# --- Rain from the corrected forecast ------------------------------------------------------


@lru_cache(maxsize=1)
def _mapping() -> dict:
    path = forecast.mapping_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m models.forecast ...")
    return json.loads(path.read_text(encoding="utf-8"))


def forecast_at_wards(day: date) -> pd.Series | None:
    """Corrected forecast rain per ward for the 24 h ending 08:30 IST on `day`."""
    fc = forecast.forecast(day)  # (model, lat, lon)
    if fc is None:
        return None
    corrected = forecast.apply_quantile_map(_mapping(), fc.mean(axis=0))
    lons, lats = target_coords()
    field = xr.DataArray(corrected, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    return flood.at_wards(field)


def forecast_issued(day: date) -> datetime | None:
    """When the forecast covering `day` was downloaded, from the raw file name."""
    path = store.clean_path("openmeteo", day)
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        found = re.search(r"\d{8}T\d{6}Z", ds.attrs.get("raw_file", ""))
    return datetime.strptime(found.group(), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC) if found else None


class RainInputs:
    """Rain per ward for each IMD window, remembering where each window came from."""

    def __init__(self, day: date, mode: str):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.day, self.mode, self.used = day, mode, {}

    def __call__(self, window: date) -> pd.Series | None:
        if self.mode == "observed" or window < self.day:
            values = flood.imd_at_wards(window)
            if values is not None:
                self.used[window] = "IMD"
                return values
            if self.mode == "observed":
                return None
        values = forecast_at_wards(window)
        if values is not None:
            self.used[window] = "forecast"
        return values

    def data_time(self) -> datetime:
        """Time of the newest input: the end of the last IMD day, or the forecast download."""
        times = []
        for window, source in self.used.items():
            if source == "IMD":
                times.append(datetime.combine(window, IMD_DAY_END))
            else:
                issued = forecast_issued(window)
                if issued:
                    times.append(issued)
        if not times:
            raise RuntimeError("no rain input was used")
        return max(times)

    def describe(self) -> str:
        return ", ".join(f"{w:%d %b} {s}" for w, s in sorted(self.used.items()))


# --- The run ------------------------------------------------------------------------------


def compute(
    day: date,
    mode: str,
    now: datetime | None = None,
    live: bool = False,
    rain=None,
    run_id: str | None = None,
    rain_source: str | None = None,
) -> dict:
    """Run the flood model for every ward. Returns the run record and validated WardRisk rows.

    `now` is the issue time. A replay passes its own `rain` provider (same interface as
    RainInputs), a fixed `run_id` and a `rain_source` label, and the simulated issue time.
    """
    now = now or datetime.now(UTC)
    model, meta = flood.load_final()
    rain = rain or RainInputs(day, mode)
    wards = pd.DataFrame([(day, w) for w in flood.ward_points().index], columns=["date", "ward_id"])
    table = flood.build(wards, rain=rain)
    missing = sorted(w for _, w in table.attrs["dropped"])
    if missing:
        raise RuntimeError(
            f"rain missing for {day} in {len(missing)} wards. Run first: python -m pipelines.run "
            f"--start {day - timedelta(days=3)} --end {day + timedelta(days=1)} "
            f"--sources {'imd' if mode == 'observed' else 'imd,openmeteo'}"
        )
    probability = model.predict_proba(table[meta["features"]])[:, 1]
    data_time = rain.data_time()
    run_id = run_id or f"flood-{mode}-{day:%Y%m%d}-{now:%Y%m%dT%H%M%SZ}"
    valid_from, valid_to = valid_window(day)
    age = max(0, int((now - data_time).total_seconds() // 60))
    risks = [
        WardRisk(
            ward_id=ward,
            hazard=Hazard.FLOOD,
            valid_from=valid_from,
            valid_to=valid_to,
            probability=round(float(p), 4),
            severity=severity(p),
            trust=Trust.LOW,
            run_id=run_id,
            data_age_minutes=age,
        )
        for ward, p in zip(table["ward_id"], probability, strict=True)
    ]
    run = {
        "run_id": run_id,
        "issued_at": now,
        "data_time": data_time,
        "rain_source": rain_source or mode,
        "model": "flood_xgb " + meta.get("created_at", ""),
        "is_test": not (live and mode == "forecast"),
        "note": f"Rain inputs: {rain.describe()}"[:500],
    }
    return {"run": run, "risks": risks, "features": table}


def save(result: dict) -> str:
    init_db()
    with SessionLocal.begin() as session:
        session.add(RiskRun(**result["run"]))
        session.flush()  # the run row must exist before the ward rows point at it
        for risk in result["risks"]:
            fields = risk.model_dump(exclude={"schema_version"})
            fields["hazard"] = risk.hazard.value
            fields["severity"] = risk.severity.value
            fields["trust"] = risk.trust.value
            session.add(WardRiskRow(**fields))
    return result["run"]["run_id"]


def list_runs() -> list[dict]:
    init_db()
    with SessionLocal() as session:
        runs = session.scalars(select(RiskRun).order_by(RiskRun.issued_at.desc())).all()
        return [
            {c: getattr(r, c) for c in ("run_id", "issued_at", "is_test", "note")} for r in runs
        ]


def delete_run(run_id: str) -> bool:
    with SessionLocal.begin() as session:
        session.execute(delete(WardRiskRow).where(WardRiskRow.run_id == run_id))
        return session.execute(delete(RiskRun).where(RiskRun.run_id == run_id)).rowcount > 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write flood risk for every ward to PostGIS.")
    parser.add_argument("--date", help="local (IST) date, YYYY-MM-DD or 'today'")
    parser.add_argument("--rain", choices=MODES, default="forecast")
    parser.add_argument(
        "--live", action="store_true", help="do not mark the run TEST (forecast mode only)"
    )
    parser.add_argument("--list", action="store_true", help="list runs in the database")
    parser.add_argument("--delete", metavar="RUN_ID")
    args = parser.parse_args(argv)

    if args.list:
        for r in list_runs():
            flag = "TEST" if r["is_test"] else "LIVE"
            print(f"{r['run_id']:<48} {flag}  issued {r['issued_at']:%Y-%m-%d %H:%M} UTC")
        return 0
    if args.delete:
        found = delete_run(args.delete)
        print(("Deleted " if found else "No run ") + args.delete)
        return 0 if found else 1
    if not args.date:
        parser.error("give --date, --list or --delete")
    day = datetime.now(IST).date() if args.date == "today" else date.fromisoformat(args.date)
    if args.live and args.rain != "forecast":
        parser.error("--live only applies to forecast runs; hindcasts are always TEST")

    try:
        result = compute(day, args.rain, live=args.live)
    except (RuntimeError, FileNotFoundError) as exc:
        print(exc)
        return 1
    run_id = save(result)
    counts = pd.Series([r.severity.value for r in result["risks"]]).value_counts()
    probs = np.array([r.probability for r in result["risks"]])
    print(f"Saved run {run_id} ({'TEST' if result['run']['is_test'] else 'LIVE'})")
    print(f"{result['run']['note']}")
    print(f"Probability {probs.min():.2f} to {probs.max():.2f}; severity {counts.to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
