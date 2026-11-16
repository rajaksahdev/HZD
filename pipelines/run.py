"""Fetch and harmonise sources for a date range. Safe to re-run: finished raw files are skipped.

python -m pipelines.run --start 2025-07-18 --end 2025-07-21
python -m pipelines.run --days-back 2 --days-ahead 2           (what cron runs)
python -m pipelines.run --start 2025-07-20 --sources chirps,imd --no-fetch
"""

import argparse
import sys
from datetime import UTC, date, datetime, timedelta

from pipelines import store
from pipelines.sources import SOURCES

FORECAST_SOURCES = {"openmeteo", "tide"}  # the only ones with data for future days


def daterange(start: date, end: date):
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)


def process(source: str, day: date, fetch: bool, force: bool) -> str:
    """Returns 'ok', 'no data' or 'error: ...' for one source and day."""
    module = SOURCES[source]
    try:
        if fetch:
            module.fetch(day, force=force)
        ds = module.harmonize(day)
        if ds is None:
            return "no data"
        ds.attrs["harmonised_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        out = store.clean_path(source, day)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        ds.to_netcdf(tmp, encoding={v: {"zlib": True} for v in ds.data_vars})
        tmp.replace(out)
        return "ok"
    except Exception as exc:  # one bad source must not stop the others
        return f"error: {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--days-back", type=int, help="start this many days before today (UTC)")
    parser.add_argument("--days-ahead", type=int, default=0, help="end this many days after today")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--no-fetch", action="store_true", help="only rebuild clean data from raw")
    parser.add_argument("--force", action="store_true", help="fetch a new copy even for past days")
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    if args.days_back is not None:
        start, end = today - timedelta(days=args.days_back), today + timedelta(days=args.days_ahead)
    elif args.start:
        start, end = args.start, args.end or args.start
    else:
        parser.error("give --start (and optionally --end), or --days-back")
    if end < start:
        parser.error("--end is before --start")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        parser.error(f"unknown sources {unknown}; choose from {sorted(SOURCES)}")

    counts = {"ok": 0, "no data": 0, "error": 0}
    for day in daterange(start, end):
        for source in sources:
            if day > today and source not in FORECAST_SOURCES:
                continue
            result = process(source, day, fetch=not args.no_fetch, force=args.force)
            counts[result.split(":")[0]] += 1
            print(f"{day}  {source:<10} {result}", flush=True)
    print(f"ok={counts['ok']}  no data={counts['no data']}  errors={counts['error']}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
