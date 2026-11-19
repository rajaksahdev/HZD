"""Replay mode: run past flood days through the risk pipeline as if live, and measure lead time.

Forecast replays (2024 onwards). Open-Meteo's Previous Runs archive keeps, for every hour, the
forecast made about 1, 2, ... 5 days earlier ("previous_dayK": made 24K to 24K+24 hours ahead).
A replay issued at time T uses, for each hour h, the forecast made at least h - T ahead, so
only model runs that already existed at T are used. IMD rain for a day counts as known once it
would have been published (assumed 12:00 IST the same day). Each flood day is replayed as if
issued 0, 1, 2 and 3 days before 00:00 IST on that day.

Observed replays (2005, 2021). The archive has no forecasts issued a day or more ahead for these
years, so they are replayed with observed IMD rain as a perfect forecast. They show the pipeline
working on a famous event but measure no lead time.

Lead time for a reported ward = the largest number of days ahead at which it was flagged in
every replay from that lead down to the day itself. Lead time means little on its own, so the
same replay is run over every monsoon day of 2024 and 2025 to count how often each lead warns.

Every replay run is saved to PostGIS as TEST, so the map can step through it.

python -m models.replay --fetch 2024-05 2025-10       download the forecast archive, by month
python -m models.replay --report docs/replay.md       replay, save runs, measure lead time
"""

import argparse
import calendar
import json
import math
import sys
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from api.config import settings
from models import flood, forecast, risk, tide
from pipelines import labels, store
from pipelines.grid import target_coords
from pipelines.sources import openmeteo

NAME = "previous_runs"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
RAW = settings.raw_dir / NAME
MAX_LAG = 5  # previous_day5: ICON keeps no older runs
VARIABLES = ["precipitation"] + [f"precipitation_previous_day{k}" for k in range(1, MAX_LAG + 1)]
IMD_PUBLISHED = timedelta(hours=3, minutes=30)  # assumed: 03:00 UTC day end + 3.5 h = 12:00 IST
LEADS = [0, 1, 2, 3]  # days before 00:00 IST on the flood day
EVENTS = {  # flood day -> how it can be replayed
    date(2005, 7, 26): "observed",  # no archived forecasts for 2005
    date(2021, 7, 18): "observed",  # 2021 archive: GFS short-range only, nothing issued days ahead
    date(2024, 7, 8): "forecast",
    date(2025, 5, 26): "forecast",
    date(2025, 8, 19): "forecast",
}
SEASON_YEARS = [2024, 2025]


# --- Archived forecasts ---------------------------------------------------------------------


def _month_file(year: int, month: int) -> Path | None:
    files = sorted((RAW / f"{year}" / f"{month:02d}").glob("previous_runs_*.json"))
    return files[-1] if files else None


def fetch_month(year: int, month: int, force: bool = False) -> Path:
    have = _month_file(year, month)
    today = datetime.now(UTC).date()
    finished = date(year, month, calendar.monthrange(year, month)[1]) < today
    if have and finished and not force:
        return have
    lats, lons = openmeteo._points()
    last = min(date(year, month, calendar.monthrange(year, month)[1]), today)
    resp = store.http_get(
        URL,
        params={
            "latitude": ",".join(f"{v:.2f}" for v in lats),
            "longitude": ",".join(f"{v:.2f}" for v in lons),
            "hourly": ",".join(VARIABLES),
            "models": ",".join(forecast.MODELS),
            "start_date": f"{year}-{month:02d}-01",
            "end_date": last.isoformat(),
            "timezone": "UTC",
        },
        timeout=300,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RAW / f"{year}" / f"{month:02d}" / f"previous_runs_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)
    store.log_download(NAME, date(year, month, 1), path, resp.url)
    return path


@lru_cache(maxsize=6)
def load_month(year: int, month: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(times, cube) for one month. cube is (lag, model, time, lat, lon) in mm per hour."""
    path = _month_file(year, month)
    if path is None:
        return None
    points = json.loads(path.read_text(encoding="utf-8"))
    points = [points] if isinstance(points, dict) else points
    lons, lats = target_coords()
    times = np.asarray(points[0]["hourly"]["time"], dtype="datetime64[h]")
    cube = np.full(
        (MAX_LAG + 1, len(forecast.MODELS), times.size, lats.size, lons.size), np.nan, "float32"
    )
    for p, point in enumerate(points):  # lat-major, as openmeteo._points()
        for lag, var in enumerate(VARIABLES):
            for m, model in enumerate(forecast.MODELS):
                values = point["hourly"].get(f"{var}_{model}")
                if values is not None:
                    cube[lag, m, :, p // lons.size, p % lons.size] = np.array(
                        [np.nan if v is None else v for v in values], dtype="float32"
                    )
    return times, cube


def lags(hours: list[datetime], issued: datetime) -> list[int]:
    """For each hour, the smallest previous_dayK whose run is certain to predate `issued`.

    previous_dayK for hour h comes from a run made 24K to 24K+24 hours before h, so it existed
    at `issued` whenever h - 24K <= issued.
    """
    return [max(0, math.ceil((h - issued) / timedelta(days=1))) for h in hours]


def window_hours(window: date) -> list[datetime]:
    """The 24 hourly steps of the IMD day ending 03:00 UTC on `window` (value = hour ending)."""
    end = datetime.combine(window, time(3, 0), tzinfo=UTC)
    return [end - timedelta(hours=23 - i) for i in range(24)]


def forecast_window(window: date, issued: datetime) -> np.ndarray | None:
    """Rain per model (model, lat, lon) for one IMD day, using only runs that existed at
    `issued`. None if the archive does not reach that far back."""
    hours = window_hours(window)
    total = None
    for h, lag in zip(hours, lags(hours, issued), strict=True):
        if lag > MAX_LAG:
            return None
        month = load_month(h.year, h.month)
        if month is None:
            return None
        times, cube = month
        idx = np.searchsorted(times, np.datetime64(h.replace(tzinfo=None), "h"))
        if idx >= times.size or times[idx] != np.datetime64(h.replace(tzinfo=None), "h"):
            return None
        step = cube[lag, :, idx]
        total = step.astype("float64") if total is None else total + step
    if np.isnan(total).any():
        return None
    return total


class ReplayRain:
    """Rain per ward as it was known at `issued`: IMD once published, else archived forecast."""

    def __init__(self, issued: datetime):
        self.issued, self.used, self._cache = issued, {}, {}

    def __call__(self, window: date) -> pd.Series | None:
        if window in self._cache:
            return self._cache[window]
        values = None
        published = datetime.combine(window, risk.IMD_DAY_END) + IMD_PUBLISHED
        if published <= self.issued:
            values = flood.imd_at_wards(window)
            if values is not None:
                self.used[window] = "IMD"
        if values is None:
            fc = forecast_window(window, self.issued)
            if fc is not None:
                corrected = forecast.apply_quantile_map(risk._mapping(), fc.mean(axis=0))
                lons, lats = target_coords()
                field = xr.DataArray(
                    corrected, coords={"lat": lats, "lon": lons}, dims=("lat", "lon")
                )
                values = flood.at_wards(field)
                self.used[window] = "forecast"
        self._cache[window] = values
        return values

    def data_time(self) -> datetime:
        times = [datetime.combine(w, risk.IMD_DAY_END) for w, s in self.used.items() if s == "IMD"]
        if any(s == "forecast" for s in self.used.values()):
            times.append(self.issued)  # the newest run used was made at or before this time
        return max(times)

    def describe(self) -> str:
        return ", ".join(f"{w:%d %b} {s}" for w, s in sorted(self.used.items()))


def issue_time(day: date, lead: int) -> datetime:
    return risk.valid_window(day)[0] - timedelta(days=lead)


# --- Lead time ------------------------------------------------------------------------------


def lead_time(flagged_by_lead: dict[int, bool]) -> int | None:
    """Largest lead N with a flag at every lead from 0 to N; None if not flagged on the day."""
    best = None
    for lead in sorted(flagged_by_lead):
        if not flagged_by_lead[lead]:
            break
        best = lead
    return best


def heavy(table: pd.DataFrame) -> np.ndarray:
    return flood.rule(table).astype(bool)


def replay_event(day: date, mode: str, save: bool = True) -> pd.DataFrame:
    """One row per ward and lead: probability, model flag, rule flag. Saves each run as TEST."""
    rows = []
    if mode == "observed":
        issued = datetime.combine(day + timedelta(days=1), risk.IMD_DAY_END) + IMD_PUBLISHED
        result = risk.compute(
            day,
            "observed",
            now=issued,
            run_id=f"replay-{day:%Y%m%d}-observed",
            rain_source="observed (replay)",
        )
        runs = [(None, result)]
    else:
        runs = []
        for lead in LEADS:
            issued = issue_time(day, lead)
            result = risk.compute(
                day,
                "forecast",
                now=issued,
                rain=ReplayRain(issued),
                run_id=f"replay-{day:%Y%m%d}-lead{lead}d",
                rain_source=f"forecast-lead{lead}d",
            )
            runs.append((lead, result))
    for lead, result in runs:
        if save:
            risk.delete_run(result["run"]["run_id"])
            risk.save(result)
        table = result["features"]
        rule_flags = heavy(table)
        for r, ward_rule in zip(result["risks"], rule_flags, strict=True):
            row = table.loc[table["ward_id"] == r.ward_id].iloc[0]
            rows.append(
                {
                    "day": day,
                    "lead": lead,
                    "ward_id": r.ward_id,
                    "probability": r.probability,
                    "severity": r.severity.value,
                    "model": r.probability >= flood.THRESHOLD,
                    "rule": bool(ward_rule),
                    "rain_overnight_mm": row["rain_overnight_mm"],
                    "rain_today_mm": row["rain_today_mm"],
                    "run_id": result["run"]["run_id"],
                    "inputs": result["run"]["note"],
                }
            )
    return pd.DataFrame(rows)


def season(year: int, model, features: list[str], tide_model: dict) -> pd.DataFrame:
    """Every monsoon day of a year at every lead: wards flagged by the model and by the rule."""
    out = []
    start, end = date(year, 6, 1), date(year, 9, 30)
    wards = flood.ward_points().index
    for n in range((end - start).days + 1):
        day = start + timedelta(days=n)
        grid = pd.DataFrame([(day, w) for w in wards], columns=["date", "ward_id"])
        for lead in LEADS:
            table = flood.build(grid, tide_model=tide_model, rain=ReplayRain(issue_time(day, lead)))
            if len(table) < len(wards):
                continue
            p = model.predict_proba(table[features])[:, 1]
            out.append(
                {
                    "day": day,
                    "lead": lead,
                    "model_wards": set(table["ward_id"][p >= flood.THRESHOLD]),
                    "rule_wards": set(table["ward_id"][heavy(table)]),
                }
            )
    return pd.DataFrame(out)


# --- Report -------------------------------------------------------------------------------


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _ahead(n: int) -> str:
    return "on the day" if n == 0 else f"{n} day{'s' * (n > 1)} ahead"


def _lead_text(flags: dict[int, bool]) -> str:
    n = lead_time(flags)
    if n is not None:
        return "on the day" if n == 0 else f"{n} day{'s' * (n > 1)}"
    early = [_ahead(k) for k in sorted(flags, reverse=True) if flags[k]]
    return f"dropped on the day (warned {', '.join(early)})" if early else "missed"


def ward_flags(table: pd.DataFrame, ward: str, column: str) -> dict[int, bool]:
    w = table[table["ward_id"] == ward].set_index("lead")
    return {n: bool(w.loc[n, column]) for n in LEADS}


def season_counts(seasons: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    """Per lead: days with any ward flagged, and reported ward-days flagged, model and rule."""
    keys = set(zip(reported["date"], reported["ward_id"], strict=True))
    rows = []
    for lead in sorted(LEADS, reverse=True):
        s = seasons[seasons["lead"] == lead]
        in_season = {k for k in keys if k[0] in set(s["day"])}
        row = {"lead": lead, "days": len(s), "reports": len(in_season)}
        for name in ("model", "rule"):
            col = s[f"{name}_wards"]
            got = {(d, w) for d, ws in zip(s["day"], col, strict=True) for w in ws}
            row[f"{name}_days"] = int((col.map(len) > 0).sum())
            row[f"{name}_caught"] = len(got & in_season)
        rows.append(row)
    return pd.DataFrame(rows)


def summary(events: dict, counts: pd.DataFrame, reported: pd.DataFrame) -> list[str]:
    flags = {"model": [], "rule": []}
    for day, table in events.items():
        if EVENTS[day] != "forecast":
            continue
        for ward in reported.loc[reported["date"] == day, "ward_id"]:
            for name in flags:
                flags[name].append(ward_flags(table, ward, name))

    def spread(all_flags: list[dict[int, bool]]) -> str:
        leads = [lead_time(f) for f in all_flags]
        parts = [
            f"{leads.count(n)} warned {_ahead(n)}" + (" or more" if n == max(LEADS) else "")
            for n in sorted(LEADS, reverse=True)
            if leads.count(n)
        ]
        dropped = sum(
            1 for f, n in zip(all_flags, leads, strict=True) if n is None and any(f.values())
        )
        never = sum(1 for f in all_flags if not any(f.values()))
        if dropped:
            parts.append(f"{dropped} warned earlier but not on the day")
        if never:
            parts.append(f"{never} never warned")
        return ", ".join(parts)

    total = len(flags["model"])
    far = counts.set_index("lead").loc[max(LEADS)]
    return [
        f"- **Measured lead time** for the {total} reported ward-days with archived forecasts "
        f"(3 flood days). Model: {spread(flags['model'])}. Heavy-rain rule: "
        f"{spread(flags['rule'])}.",
        f"- **The price of that lead time.** Replayed {_ahead(max(LEADS))} over every monsoon "
        f"day of {SEASON_YEARS[0]}-{SEASON_YEARS[-1]}, the model flags at least one ward on "
        f"{far['model_days']} of {far['days']} days ({far['model_days'] / far['days']:.0%}) and "
        f"the rule on {far['rule_days']} ({far['rule_days'] / far['days']:.0%}). The model's "
        "lead time comes from warning on most rainy days, so it is not yet useful. The rule "
        f"caught {far['rule_caught']} of {far['reports']} reported monsoon ward-days "
        f"{_ahead(max(LEADS))} with far fewer warnings.",
        "- **Observed replays** (26 July 2005, 18 July 2021) flag every reported ward, as they "
        "must with perfect rain, and measure no lead time.",
    ]


def report(events: dict, seasons: pd.DataFrame, reported: pd.DataFrame, command: str) -> str:
    lines = [
        "# Replay: past flood days run as if live",
        "",
        f"Generated by `{command}`. Do not edit by hand. Every run here is saved to the "
        "database as TEST; pick it in the map's run list or open `/?run=<run id>`.",
        "",
        "## Result",
        "",
        *summary(events, season_counts(seasons, reported), reported),
        "",
        "## How a replay works",
        "",
        "- **Forecast replays (2024, 2025):** issued at 00:00 IST, 0 to 3 days before the flood "
        "day. Each hour's rain comes from an Open-Meteo archived forecast made at least that far "
        "ahead (Previous Runs archive: the forecast made 1 to 5 days before each hour), "
        "bias-corrected with the Stage 4 quantile map. IMD rain for a day counts as known from "
        "12:00 IST that day (assumed publication time).",
        "- **Observed replays (2005, 2021):** no forecast made a day or more ahead is archived "
        "for these years (the archive starts in 2024; for 2021 it holds GFS short-range "
        "forecasts only). They use observed IMD rain as a perfect forecast, so they measure no "
        "lead time.",
        "- **Flagged** means the Stage 5 model gives 0.5 or more (orange or red on the map). "
        "The heavy-rain rule flags 64.5 mm or more in either 24 h window.",
        "- **Lead time** for a reported ward is the largest number of days ahead at which it was "
        "flagged in every replay from that lead down to the day itself. Days are counted from "
        "00:00 IST on the reported date; the flooding itself came some hours later.",
        "",
    ]
    for day, table in events.items():
        wards = sorted(reported.loc[reported["date"] == day, "ward_id"])
        mode = EVENTS[day]
        lines += [f"## {day:%d %B %Y} ({mode} rain)", ""]
        lines.append(f"Wards with a flood report: {', '.join(wards)}.")
        lines.append("")
        if mode == "observed":
            t = table.set_index("ward_id")
            flagged = [w for w in wards if t.loc[w, "model"]]
            rule = [w for w in wards if t.loc[w, "rule"]]
            lines += [
                f"Run `{table['run_id'].iloc[0]}`. IMD rain at the reported wards: "
                f"{t.loc[wards, 'rain_overnight_mm'].min():.0f}-"
                f"{t.loc[wards, 'rain_overnight_mm'].max():.0f} mm before 08:30 IST, "
                f"{t.loc[wards, 'rain_today_mm'].min():.0f}-"
                f"{t.loc[wards, 'rain_today_mm'].max():.0f} mm after.",
                "",
                f"- Model flags {len(flagged)} of {len(wards)} reported wards; all wards flagged: "
                f"{int(table['model'].sum())} of 24.",
                f"- Heavy-rain rule flags {len(rule)} of {len(wards)} reported wards; all wards "
                f"flagged: {int(table['rule'].sum())} of 24.",
                "- Lead time: not measurable (no archived forecasts).",
                "",
            ]
            continue
        lines += [
            _row(
                [
                    "Ward",
                    *[_ahead(n).capitalize() for n in sorted(LEADS, reverse=True)],
                    "Model lead time",
                    "Rule lead time",
                ]
            ),
            "|" + "---|" * (len(LEADS) + 3),
        ]
        for ward in wards:
            w = table[table["ward_id"] == ward].set_index("lead")
            cells = [
                f"{w.loc[n, 'probability']:.2f}{' R' if w.loc[n, 'rule'] else ''}"
                for n in sorted(LEADS, reverse=True)
            ]
            lines.append(
                _row(
                    [
                        ward,
                        *cells,
                        _lead_text(ward_flags(table, ward, "model")),
                        _lead_text(ward_flags(table, ward, "rule")),
                    ]
                )
            )
        rain = table[table["ward_id"].isin(wards)].groupby("lead")["rain_today_mm"].mean()
        lines += [
            "",
            "Cells: model probability; R = heavy-rain rule flag. Corrected forecast rain for the "
            "day (after 08:30 IST, mean over reported wards): "
            + ", ".join(f"{rain[n]:.0f} mm {_ahead(n)}" for n in sorted(LEADS, reverse=True))
            + f". IMD observed: {_observed_today(day, wards):.0f} mm.",
            "",
        ]
    lines += [
        "## How often it would warn at each lead",
        "",
        "Every day of June to September 2024 and 2025, replayed at each lead. A day counts when "
        "at least one ward is flagged. Reported ward-days are the flood reports in those seasons "
        "(a ward with no report is unknown, not dry).",
        "",
        _row(
            [
                "Lead",
                "Days replayed",
                "Model: days flagged",
                "Model: reports flagged",
                "Rule: days flagged",
                "Rule: reports flagged",
            ]
        ),
        "|---|---|---|---|---|---|",
    ]
    for r in season_counts(seasons, reported).to_dict("records"):
        lines.append(
            _row(
                [
                    _ahead(r["lead"]),
                    r["days"],
                    r["model_days"],
                    f"{r['model_caught']} of {r['reports']}",
                    r["rule_days"],
                    f"{r['rule_caught']} of {r['reports']}",
                ]
            )
        )
    lines += [
        "",
        "## Limits",
        "",
        "- Five flood days, three of them with archived forecasts. These are examples, not a "
        "skill estimate.",
        "- The quantile map was learned on short-range forecasts; it is applied unchanged to "
        "forecasts made days ahead.",
        "- The exact start time of each archived run is not given, only that it was made 24K to "
        "24K+24 hours ahead, so a replay may use a run a few hours older than a live system "
        "would have had.",
        "- The model inherits the Stage 5 problem: it flags nearly any rainy day, so long lead "
        "times come with many warning days. Compare the model and rule columns, not lead time "
        "alone.",
        "",
    ]
    return "\n".join(lines)


def _observed_today(day: date, wards: list[str]) -> float:
    values = flood.imd_at_wards(day + timedelta(days=1))
    return float("nan") if values is None else float(values[wards].mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay past flood days as if live.")
    parser.add_argument("--fetch", nargs=2, metavar=("FROM", "TO"), help="months, YYYY-MM")
    parser.add_argument("--report", type=Path, help="replay and write a Markdown report here")
    parser.add_argument("--no-save", action="store_true", help="do not write runs to PostGIS")
    args = parser.parse_args(argv)
    if not args.fetch and not args.report:
        parser.error("give --fetch and/or --report")

    if args.fetch:
        first, last = (pd.Period(v, "M") for v in args.fetch)
        for period in pd.period_range(first, last, freq="M"):
            path = fetch_month(period.year, period.month)
            print(f"{period}  {path.relative_to(settings.raw_dir).as_posix()}")

    if args.report:
        table = labels.ward_days(labels.read())
        reported = table[table["flooded"] == 1]
        events = {}
        for day, mode in EVENTS.items():
            events[day] = replay_event(day, mode, save=not args.no_save)
            print(f"replayed {day} ({mode}): {events[day]['run_id'].nunique()} runs")
        model, meta = flood.load_final()
        tide_model = tide.load_model()
        seasons = pd.concat(
            [season(y, model, meta["features"], tide_model) for y in SEASON_YEARS],
            ignore_index=True,
        )
        command = "python -m models.replay " + " ".join(argv if argv is not None else sys.argv[1:])
        args.report.write_text(report(events, seasons, reported, command), encoding="utf-8")
        print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
