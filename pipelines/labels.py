"""Flood labels: one row per reported (or ruled-out) flooding of a ward on a date.

The table is labels/flood_events.csv. It is edited by hand and kept in git, because every row
must be traceable to a source. A "ward-day" is one ward on one date.

Columns
  date         local (IST) calendar date
  ward_id      one of the 24 ward ids, e.g. F-N
  flooded      1 = waterlogging reported in the ward, 0 = no flooding
  location     the place named in the source
  lat, lon     approximate point for that place, used only to check the ward
  evidence     news | bmc_list | official | dry_day
  source_url   where a person can check the row (required when flooded = 1)
  confidence   medium = the page was opened and names the place
               low    = taken from a search summary, or the place name is vague
  reviewed     yes once a team member has opened the source and confirmed the row
  notes

dry_day rows are not observations. They are days when the IMD grid shows under 2.5 mm over the
whole pilot area on that day, the day before and the day after, so rain flooding is ruled out.
They are easy negatives: they teach a model little about heavy-rain days that did not flood.

python -m pipelines.labels                  validate the table and print a summary
python -m pipelines.labels --dry-days 2024  list dry monsoon days in a year (downloads IMD data)
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from api.config import settings
from scripts.load_wards import PATH as WARDS_PATH
from scripts.load_wards import ward_id_from_name

PATH = Path(__file__).resolve().parents[1] / "labels" / "flood_events.csv"
COLUMNS = [
    "date",
    "ward_id",
    "flooded",
    "location",
    "lat",
    "lon",
    "evidence",
    "source_url",
    "confidence",
    "reviewed",
    "notes",
]
WARD_IDS = set(
    "A B C D E F-N F-S G-N G-S H-E H-W K-E K-W L M-E M-W N P-N P-S R-C R-N R-S S T".split()
)
EVIDENCE = {"news", "bmc_list", "official", "dry_day"}
CONFIDENCE = {"medium", "low", "high"}
DRY_MM = 2.5  # IMD counts under 2.5 mm in a day as no rain or very light rain
MIN_WARD_DAYS = 30


def read(path: Path = PATH) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return table[COLUMNS]


def problems(table: pd.DataFrame, wards: gpd.GeoDataFrame | None = None) -> list[str]:
    """Everything wrong with the table, one message per problem. Empty list = valid."""
    found = []
    for n, row in enumerate(table.to_dict("records"), start=2):  # line 1 is the header

        def bad(message: str, n=n) -> None:
            found.append(f"line {n}: {message}")

        try:
            date.fromisoformat(row["date"])
        except ValueError:
            bad(f"date {row['date']!r} is not YYYY-MM-DD")
        if row["ward_id"] not in WARD_IDS:
            bad(f"unknown ward_id {row['ward_id']!r}")
        if row["flooded"] not in {"0", "1"}:
            bad(f"flooded must be 0 or 1, got {row['flooded']!r}")
        if row["evidence"] not in EVIDENCE:
            bad(f"evidence must be one of {sorted(EVIDENCE)}")
        if row["confidence"] not in CONFIDENCE:
            bad(f"confidence must be one of {sorted(CONFIDENCE)}")
        if row["reviewed"] not in {"yes", "no"}:
            bad("reviewed must be yes or no")
        if row["flooded"] == "1" and not row["source_url"].startswith("http"):
            bad("a flooded row needs a source_url")
        if row["flooded"] == "1" and row["evidence"] == "dry_day":
            bad("a dry_day row cannot be flooded")
        if bool(row["lat"]) != bool(row["lon"]):
            bad("give both lat and lon, or neither")
        elif row["lat"] and wards is not None and row["ward_id"] in WARD_IDS:
            try:
                point = Point(float(row["lon"]), float(row["lat"]))
            except ValueError:
                bad("lat and lon must be numbers")
                continue
            inside = wards.loc[wards.contains(point), "ward_id"].tolist()
            if inside != [row["ward_id"]]:
                bad(f"the point is in ward {inside or 'none'}, not {row['ward_id']}")
    days = table.groupby(["date", "ward_id"])["flooded"].nunique()
    for (day, ward), count in days.items():
        if count > 1:
            found.append(f"{day} {ward}: marked both flooded and not flooded")
    return found


def ward_days(table: pd.DataFrame) -> pd.DataFrame:
    """One row per ward-day: date, ward_id, flooded (0/1), and how many source rows back it."""
    out = table.groupby(["date", "ward_id", "flooded"]).size().rename("rows").reset_index()
    out["flooded"] = out["flooded"].astype(int)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def load_wards() -> gpd.GeoDataFrame | None:
    if not WARDS_PATH.exists():
        return None
    wards = gpd.read_file(WARDS_PATH)
    wards["ward_id"] = wards["name"].map(ward_id_from_name)
    return wards


def dry_days(year: int) -> list[date]:
    """Monsoon days (June to September) with under DRY_MM over the pilot area for three days."""
    from pipelines.sources import imd

    imd.fetch(date(year, 7, 1))
    src = imd._open(date(year, 7, 1))
    if src is None:
        raise RuntimeError(f"No IMD rainfall file for {year}")
    min_lon, min_lat, max_lon, max_lat = settings.pilot_bbox
    box = src["rain"].sel(lon=slice(min_lon - 0.25, max_lon + 0.25))
    box = box.sel(lat=slice(min_lat - 0.25, max_lat + 0.25))
    daily = box.where(box >= 0).max(dim=["lat", "lon"]).to_series()
    daily.index = pd.to_datetime(daily.index).date
    # IMD's value for a date is the 24 h ending 08:30 IST that morning, so a calendar day is
    # covered by its own value and the next day's. Add the day before as a buffer.
    window = pd.concat([daily.shift(1), daily, daily.shift(-1)], axis=1).max(axis=1, skipna=False)
    dry = window[window < DRY_MM]
    return [d for d in dry.index if 6 <= d.month <= 9]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and summarise the flood label table.")
    parser.add_argument("--dry-days", type=int, metavar="YEAR", help="list dry monsoon days")
    args = parser.parse_args(argv)
    if args.dry_days:
        days = dry_days(args.dry_days)
        print(f"{len(days)} dry monsoon days in {args.dry_days}:")
        print(" ".join(d.isoformat() for d in days))
        return 0

    table = read()
    wards = load_wards()
    if wards is None:
        print("Ward file not found, so points are not checked. Run: python -m scripts.load_wards")
    found = problems(table, wards)
    for message in found:
        print(message)
    if found:
        print(f"{len(found)} problems in {PATH}")
        return 1

    days = ward_days(table)
    flooded = days[days["flooded"] == 1]
    not_flooded = days[days["flooded"] == 0]
    reviewed = int((table["reviewed"] == "yes").sum())
    print(f"{len(table)} rows, {reviewed} reviewed by a person")
    print(f"{len(days)} ward-days: {len(flooded)} flooded, {len(not_flooded)} not flooded")
    print(f"{days['date'].nunique()} dates, {days['ward_id'].nunique()} of 24 wards")
    print("\nFlooded ward-days by date:")
    by_date = flooded.groupby("date")["ward_id"].apply(lambda s: " ".join(sorted(s)))
    for day, names in by_date.items():
        print(f"  {day}  {names}")
    print("\nRows by evidence and confidence:")
    print(table.groupby(["evidence", "confidence"]).size().to_string())
    enough = len(days) >= MIN_WARD_DAYS and len(flooded) and len(not_flooded)
    verdict = "met" if enough else "not met"
    print(f"\nTarget of {MIN_WARD_DAYS} flood and no-flood ward-days: {verdict}")
    return 0 if enough else 1


if __name__ == "__main__":
    sys.exit(main())
