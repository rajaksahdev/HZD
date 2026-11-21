"""The demo, end to end, from a fresh clone (after `bash start.sh`).

python -m scripts.demo                  build what is missing, then check the API and map
python -m scripts.demo --serve          ... and start the map at http://localhost:8000
python -m scripts.demo --force          rebuild every step
python -m scripts.demo --skip-replay    leave out the replay step (the slowest download)

Steps, in order (each is the README command for its stage; a step whose output exists is
skipped, so an interrupted run resumes):

  static    ward terrain, drains and land cover (Stage 3)
  rain      IMD rain for every label year, Open-Meteo forecasts for the 2024-2025 seasons
  tide      harmonic tide fitted on 2023-2024 (Stage 5)
  forecast  rain bias correction learned on 2024, scored on 2025 (Stage 4)
  flood     XGBoost flood model (Stage 5)
  heat      ERA5 1991-2025, full-year forecasts 2024-2025, heat levels and correction (Stage 9)
  blend     archived forecasts 2021-2025 from six models, blend and trust levels (Stage 11),
            before today's runs so they carry trust
  today     today's data, today's flood and heat runs, and two hindcasts: flood on
            19 Aug 2025, heat on 23 Oct 2025 (Stages 6 and 9)
  replay    archived forecasts and the five replayed flood days (Stage 7)
  landslide landslide catalog, terrain for MMR and Raigad, susceptibility model (Stage 10)
  impact    population, hospitals, shelter candidates, roads; vulnerability per ward and routes
            for the hindcast (Stage 12)
  check    the API answers for every ward and the map is built

Reports from the demo go to data/demo/, not docs/, so a demo run never changes committed files.
Needs no accounts. Every run it saves is marked TEST: the demo never passes --live.
"""

import argparse
import subprocess
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from api.config import ROOT, settings

PY = sys.executable
OUT = settings.data_dir / "demo"
MODELS = settings.processed_dir / "models"
HINDCAST_DAY = "2025-08-19"
HEAT_DAY = "2025-10-23"  # the hottest day of 2025 in ERA5
HEAT_DATA = (date(2024, 1, 31), date(2025, 12, 31))  # forecasts: ECMWF archive from Feb 2024
HEAT_SPLIT = (
    ("--train", "2024-02-01", "2024-07-31"),
    ("--validate", "2024-08-01", "2024-12-31"),
    ("--test", "2025-01-01", "2025-12-31"),
)
FORECAST_SEASONS = (2024, 2025)  # Stage 4: train on 2024, test on 2025
BLEND_YEARS = (2021, 2025)  # Stage 11: the forecast archive has two or more models from 2021
RETRY_WAIT = 60  # seconds before retrying a failed step
REPLAY_MONTHS = (("2024-05", "2024-10"), ("2025-05", "2025-10"))


def label_years() -> list[int]:
    table = pd.read_csv(ROOT / "labels" / "flood_events.csv", usecols=["date"])
    return sorted(pd.to_datetime(table["date"]).dt.year.unique().tolist())


def processed(source: str, day: date) -> Path:
    return settings.processed_dir / source / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.nc"


def run(*args: str) -> None:
    command = [PY, "-m", *args]
    print("  $ python -m " + " ".join(args), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


# --- Steps: (name, done?, do) ---------------------------------------------------------------


def static_done() -> bool:
    return (settings.processed_dir / "static" / "ward_features.csv").exists()


def static() -> None:
    run("pipelines.static")


def rain_jobs() -> list[tuple[list[str], Path]]:
    """(pipelines.run arguments, a file that exists once the job is done)."""
    blend_years = range(BLEND_YEARS[0], BLEND_YEARS[1] + 1)
    years = sorted(set(label_years()) | set(FORECAST_SEASONS) | set(blend_years))
    jobs = []
    for year in years:
        start, end = date(year, 5, 15), date(year, 10, 5)
        args = ["--start", start.isoformat(), "--end", end.isoformat(), "--sources", "imd"]
        jobs.append((args, processed("imd", end)))
    for year in FORECAST_SEASONS:
        start, end = date(year, 5, 30), date(year, 10, 1)
        args = ["--start", start.isoformat(), "--end", end.isoformat(), "--sources", "openmeteo"]
        jobs.append((args, processed("openmeteo", end)))
    return jobs


def rain_done() -> bool:
    return all(path.exists() for _, path in rain_jobs())


def rain(force: bool = False) -> None:
    for args, path in rain_jobs():
        if force or not path.exists():
            run("pipelines.run", *args)


def tide_done() -> bool:
    return (MODELS / "tide_harmonics.json").exists()


def tide() -> None:
    run("models.tide", "--fit", "2023-01-01", "2024-12-31", "--check", "2025-01-01", "2025-10-31")


def forecast_done() -> bool:
    return (MODELS / "rain_quantile_map.json").exists()


def forecast() -> None:
    run(
        "models.forecast",
        *("--train", f"{FORECAST_SEASONS[0]}-06-01", f"{FORECAST_SEASONS[0]}-09-30"),
        *("--test", f"{FORECAST_SEASONS[1]}-06-01", f"{FORECAST_SEASONS[1]}-09-30"),
        *("--report", (OUT / "forecast_skill.md").as_posix()),
    )


def flood_done() -> bool:
    return (MODELS / "flood_xgb.json").exists() and (MODELS / "flood_xgb_meta.json").exists()


def flood() -> None:
    run("models.flood", "--test-from", "2024", "--report", (OUT / "flood_model.md").as_posix())


def heat_done() -> bool:
    return (MODELS / "heat.json").exists()


def heat() -> None:
    first, last = (d.isoformat() for d in HEAT_DATA)
    run("pipelines.run", "--start", first, "--end", last, "--sources", "openmeteo")
    run("models.heat", "--fetch", "1991", "2025")
    split = [part for period in HEAT_SPLIT for part in period]
    run("models.heat", *split, "--report", (OUT / "heat.md").as_posix())


def today() -> None:  # always runs: today changes every day
    run("pipelines.run", "--days-back", "4", "--days-ahead", "1", "--sources", "imd,openmeteo")
    run("models.risk", "--date", "today", "--rain", "forecast")
    run("models.risk", "--date", HINDCAST_DAY, "--rain", "observed")
    run("models.heat", "--date", "today")
    run("models.heat", "--date", HEAT_DAY, "--source", "era5")


def landslide_done() -> bool:
    return (MODELS / "landslide_xgb.json").exists()


def landslide() -> None:
    run("models.landslide", "--fetch")
    run("models.landslide", "--report", (OUT / "landslide.md").as_posix())


def impact_done() -> bool:
    return (settings.processed_dir / "impact" / "ward_vulnerability.csv").exists()


def impact() -> None:
    from models import impact as stage12
    from models import risk

    run("models.impact", "--fetch")
    run("models.impact", "--build", "--report", (OUT / "impact.md").as_posix())
    hindcast = f"flood-observed-{HINDCAST_DAY.replace('-', '')}"
    runs = [r["run_id"] for r in risk.list_runs() if r["run_id"].startswith(hindcast)]
    replays = [r["run_id"] for r in risk.list_runs() if r["run_id"] == "replay-20250819-lead3d"]
    if runs[:1] + replays and stage12.inputs_ready():
        run("models.impact", "--routes", *(runs[:1] + replays))


def blend_done() -> bool:
    return (MODELS / "blend.json").exists()


def blend() -> None:  # months already downloaded are skipped by --fetch itself
    run("models.blend", "--fetch", *(str(y) for y in BLEND_YEARS))
    run("models.blend", "--report", (OUT / "blend.md").as_posix())


def replay() -> None:  # months already downloaded are skipped by --fetch itself
    for first, last in REPLAY_MONTHS:
        run("models.replay", "--fetch", first, last)
    run("models.replay", "--report", (OUT / "replay.md").as_posix())


def check() -> list[str]:
    """Ask the API what the map would ask. Returns the problems found (empty when all is well)."""
    import warnings

    warnings.filterwarnings("ignore", message=".*httpx.*")  # starlette's test-client notice
    from fastapi.testclient import TestClient

    from api import main

    problems = []
    client = TestClient(main.app)
    if client.get("/v1/health").json().get("status") != "ok":
        return ["/v1/health: database not reachable"]
    body = client.get("/v1/wards").json()
    wards = [f["properties"] for f in body["features"]]
    coloured = [w for w in wards if "severity" in w]
    run_id = (body.get("run") or {}).get("run_id")
    print(f"  map run: {run_id}, {len(coloured)} of {len(wards)} wards coloured")
    if len(wards) != 24:
        problems.append(f"/v1/wards returned {len(wards)} wards, expected 24")
    if len(coloured) != 24:
        problems.append(f"the current run colours {len(coloured)} wards, expected 24")
    for warning in body.get("warnings", []):
        print(f"  warning shown: {warning}")
    point = client.get("/v1/risk", params={"lat": 19.0178, "lon": 72.8426})
    if point.status_code != 200 or point.json()["ward_id"] != "G-N":
        problems.append(f"/v1/risk for Dadar did not return ward G-N ({point.status_code})")
    runs = {r["run_id"] for r in client.get("/v1/runs", params={"limit": 200}).json()}
    hindcast = f"flood-observed-{HINDCAST_DAY.replace('-', '')}"
    if not any(r.startswith(hindcast) for r in runs):
        problems.append(f"no hindcast run for {HINDCAST_DAY}")
    heat = client.get("/v1/wards", params={"hazard": "heat"}).json()
    hot = [f["properties"] for f in heat["features"] if f["properties"].get("value") is not None]
    print(f"  heat run: {(heat.get('run') or {}).get('run_id')}, {len(hot)} wards with a value")
    if len(hot) != 24:
        problems.append(f"the current heat run has a value for {len(hot)} wards, expected 24")
    people = client.get("/v1/impact").json()
    with_people = [w for w in people.get("wards", []) if w.get("population")]
    with_shelter = [w for w in with_people if (w.get("shelter") or {}).get("facility_name")]
    print(f"  impact: {len(with_people)} wards with population, {len(with_shelter)} with a shelter")
    if len(with_people) != 24 or len(with_shelter) != 24:
        problems.append("Stage 12: expected population and a nearest shelter for all 24 wards")
    if not (ROOT / "web" / "dist" / "index.html").exists():
        problems.append("map not built: cd web && npm ci && npm run build")
    elif client.get("/").status_code != 200:
        problems.append("the API does not serve the built map at /")
    return problems


STEPS = [
    ("static", static_done, static),
    ("rain", rain_done, rain),
    ("tide", tide_done, tide),
    ("forecast", forecast_done, forecast),
    ("flood", flood_done, flood),
    ("heat", heat_done, heat),
    ("blend", blend_done, blend),
    ("today", lambda: False, today),
    ("replay", lambda: False, replay),
    ("landslide", landslide_done, landslide),
    ("impact", impact_done, impact),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the demo end to end.")
    parser.add_argument("--force", action="store_true", help="rebuild every step")
    parser.add_argument("--skip-replay", action="store_true", help="leave out Stage 7")
    parser.add_argument("--serve", action="store_true", help="start the map when done")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    began = time.monotonic()
    print(f"HazardLens demo, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. All runs are TEST.")
    for number, (name, done, do) in enumerate(STEPS, start=1):
        header = f"[{number}/{len(STEPS) + 1}] {name}"
        if name == "replay" and args.skip_replay:
            print(f"{header}: skipped (--skip-replay)")
            continue
        if done() and not args.force:
            print(f"{header}: already built")
            continue
        print(f"{header}", flush=True)
        step_began = time.monotonic()
        for attempt in (1, 2):  # every step resumes, so one retry rides out a dropped download
            try:
                if name == "rain":
                    rain(force=args.force and attempt == 1)
                else:
                    do()
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    print(f"\nDemo FAILED at step {name}. Run the demo again to resume.")
                    return 1
                print(f"  step {name} failed; retrying once in {RETRY_WAIT} s", flush=True)
                time.sleep(RETRY_WAIT)
        print(f"  done in {time.monotonic() - step_began:.0f} s")

    print(f"[{len(STEPS) + 1}/{len(STEPS) + 1}] check")
    problems = check()
    minutes = (time.monotonic() - began) / 60
    if problems:
        print("\nDemo FAILED:\n  " + "\n  ".join(problems))
        return 1
    print(f"\nDemo OK in {minutes:.1f} min. Reports in {OUT.relative_to(ROOT).as_posix()}/.")
    print(f"Map: uvicorn api.main:app --port {args.port}   then http://localhost:{args.port}")
    print(f"     a flood day: http://localhost:{args.port}/?run=replay-20250819-lead3d")
    if args.serve:
        subprocess.run(
            [PY, "-m", "uvicorn", "api.main:app", "--port", str(args.port)], cwd=ROOT, check=False
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
