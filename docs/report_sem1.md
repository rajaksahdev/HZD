# HazardLens: Semester 1 report

Rain and flood risk map for Mumbai's 24 wards. Stages 1 to 8 of the project plan.
Written 19 November 2026.

HazardLens is a decision-support prototype. Official warnings come from IMD, NDMA and the
Maharashtra state authorities. Every run the prototype has made is marked TEST.

## Summary

- **The pipeline runs end to end.** One command (`python -m scripts.demo`) builds the
  semester's work from a fresh clone: it downloads the data, trains the models, saves today's
  ward risk and the replayed flood days to PostGIS, and checks that the API and map serve them.
  It needs no accounts. Tested on a fresh clone with its own empty database: about 17 minutes
  and 225 MB of downloads after `start.sh`, and the forecast-skill, flood-model and replay
  reports it wrote were identical to the committed ones.
- **Rain forecasts are useful.** After bias correction, the three-model average catches 55%
  of heavy-rain cell-days (64.5 mm or more) a day ahead, against 40% for persistence.
- **The flood model is not yet better than a simple heavy-rain rule.** Both catch every
  reported flood, but over a whole monsoon the model warns on about two days in three and the
  rule on one in ten. The cause is the labels: they contain no heavy-rain day on which a ward
  stayed dry, so the model learns only "it rained". Use the rule until the labels improve.
- **Replays give a measured lead time.** On three 2024-2025 flood days run as if live from
  archived forecasts, the heavy-rain rule warned 10 of 16 flooded ward-days 3 days ahead and
  missed 4 (8 July 2024, when forecasts gave 16-37 mm and 116 mm fell).
- **The biggest gap is the labels.** None of the 67 label rows has been checked by a person.
  Reviewing them and adding BMC waterlogging lists is the first task for Semester 2.

## Stages against the plan

| Stage | Done when (plan) | Status | Evidence |
|---|---|---|---|
| 1. Ingestion | One command rebuilds clean data for any date range | Met, 4 of 8 sources | `pipelines/run.py`; IMERG, GSMaP_NOW, ERA5 and MOSDAC need accounts |
| 2. Validation and QC | Detection rate on injected faults is reported | Met | `docs/qc_report.md` |
| 3. Labels and static features | At least 30 dated flood and no-flood ward-days | Met on count, unreviewed | 59 ward-days; `docs/labels.md` |
| 4. Forecast core | Skill table (CSI, POD, FAR) against baselines | Met without pySTEPS | `docs/forecast_skill.md` |
| 5. Flood model | Metrics on held-out years; false-alarm rate stated | Met; not better than the rule | `docs/flood_model.md` |
| 6. API and map | Map shows ward risk from the database | Met | `api/main.py`, `web/` |
| 7. Replay mode | Replay demo with measured lead time | Met, with 2024-2025 in place of 2021 | `docs/replay.md` |
| 8. Wrap-up | Fresh clone runs the demo end to end | Met | `scripts/demo.py`, this report |

## What was built

### 1. Ingestion

One fetcher per source writes untouched downloads to `data/raw/{source}/YYYY/MM/DD/`, each logged
with its checksum in `data/raw/manifest.jsonl`. The clean zone holds one NetCDF file per source
and day on a 0.1 degree pilot grid (30 cells), with time in UTC.

| Source | Status |
|---|---|
| Open-Meteo forecast (ECMWF, GFS, ICON) | built |
| Open-Meteo marine sea level | built (from 2023 only) |
| IMD gridded daily rain, 0.25 degree | built (yearly files and the real-time grid) |
| CHIRPS v3 | built (about a month behind) |
| Open-Meteo Previous Runs (archived forecasts) | built in Stage 7 (from 2024 only) |
| IMERG, GSMaP_NOW, ERA5, MOSDAC | not built: each needs an account |

`python -m pipelines.run --start A --end B` rebuilds any range, and `--days-back 2 --days-ahead 2`
is the hourly scheduled run (cron or Windows Task Scheduler).

### 2. Validation and QC

Every dataset passes schema, range and gap checks before it reaches the clean zone. An Isolation
Forest was trained on hourly forecast and sea-level series and tested on injected faults (12
series-and-size combinations, 60 events each):

| Fault | Detection rate | Window already flagged without the fault |
|---|---|---|
| Spike | 91% | 2% |
| Flat line | 81% | 11% |
| Drift over 48 h | 88% | 21% |

Faults of 4 standard deviations or more are almost always caught; 2-sigma faults about half the
time. A flat line in wind speed is the hardest case (27% at 6 hours). The detector is not applied
to live data, because there are no station feeds to apply it to.

### 3. Labels and static features

- **Static features** for each ward from Copernicus DEM, ESA WorldCover and OpenStreetMap
  waterways: area, elevation, share of land below 5 m, slope, distance to a mapped waterway,
  waterway density, built-up, green, water and wetland shares.
- **Labels** (`labels/flood_events.csv`): 59 ward-days, 37 flooded and 22 not flooded, on 8
  flood dates from 2005 to 2025. Flooded rows come from news reports collected by an AI
  assistant; not-flooded rows are dry monsoon days. No row has been reviewed by a person.

### 4. Forecast core

Daily rain from ECMWF, GFS and ICON, their average, and the average after quantile-mapping bias
correction, scored against IMD rain for June-September 2025 (trained on 2024). Forecasts are up
to about a day ahead.

| Threshold | Corrected mean CSI | Best single model CSI | Persistence CSI |
|---|---|---|---|
| 2.5 mm | 0.86 | 0.88 (ICON) | 0.82 |
| 15.6 mm | 0.58 | 0.59 (ECMWF) | 0.48 |
| 64.5 mm (heavy) | 0.45 | 0.38 (ECMWF) | 0.25 |
| 115.6 mm (very heavy) | 0.48 | 0.35 (ECMWF) | 0.16 |

Bias correction matters most for heavy rain: the plain average catches 1 of 91 very heavy
cell-days and the corrected average 71, at the cost of more false alarms (FAR 0.45). The 90%
ranges for heavy rain are wide (CSI 0.30 to 0.60) because one season has few heavy days. The
pySTEPS nowcast baseline was not run: it needs IMERG or GSMaP rain maps.

### 5. Flood model

XGBoost on IMD rain around the day, the day's high tide (a harmonic tide model, off by 0.10 m
on 2025 high tides) and the ward features, with physical monotone constraints. Split by time:
trained before 2024, tested on 2024-2025.

- On held-out years the model and the heavy-rain rule both score perfectly (16 of 16 floods, no
  false alarms). This says little: every not-flooded test row is a dry day.
- The honest test is a whole monsoon: the model flags at least one ward on 81 (2024) and 80
  (2025) of 122 days; the rule on 31 and 24. Both flag every reported flood.
- The final model uses only overnight rain, cut at 8.9 mm. Tide and ward features add nothing
  until the labels contain heavy-rain days without flooding.

### 6. API and map

- `models/risk.py` runs the flood model for every ward for one IST day and saves the run to
  PostGIS (`risk_runs`, `ward_risk`). Forecast mode uses IMD where published and the corrected
  forecast for the rest.
- FastAPI: `/v1/risk?lat=&lon=`, `/v1/wards/{id}/risk`, `/v1/wards` (GeoJSON), `/v1/runs`,
  `/v1/health`. Every answer carries the data age and on-screen warnings: TEST, out of date,
  stale input (over 3 hours), wards at orange or red, low trust.
- React and MapLibre map: wards coloured by severity, a red banner for orange or red wards, data
  age, a run picker, and a click anywhere for that place's ward and risk.

Severity bands (green below 0.25, yellow, orange from 0.5, red from 0.75) are provisional until
the alert rules of Stage 13.

### 7. Replay mode

Past flood days re-run as if live: for each hour, only a forecast that already existed at the
issue time is used, and IMD rain only once it would have been published.

| | Model | Heavy-rain rule |
|---|---|---|
| Flooded ward-days warned 3 days ahead (of 16) | 16 | 10 |
| Warned earlier, dropped on the day | 0 | 2 |
| Never warned | 0 | 4 |
| Monsoon days with a warning, 3 days ahead (2024-2025) | 161 of 244 (66%) | 24 of 244 (10%) |

### 8. Integration

- `scripts/demo.py` runs every stage in order and skips steps already built, so an interrupted
  run resumes. Reports from a demo run go to `data/demo/`, so it never changes committed files.
- The demo ends with checks against the API: all 24 wards drawn and coloured, a point in Dadar
  answered with ward G-N, the 19 August 2025 hindcast present, the built map served at `/`.
- The fresh-clone test found two faults, both fixed: the IMD server often drops the first
  connection (the fetcher now retries and deletes a half-written file), and IMD serves empty
  files for days not yet published (now counted as "no data", and the risk run uses the
  forecast for those days, as designed).
- The test suite (`pytest`) has 79 tests; the database tests skip themselves when PostGIS is
  not running.

## Changes from the plan

- **Replay of July 2021 from archived forecasts was not possible.** The only free archive of
  forecasts made days ahead (Open-Meteo Previous Runs) starts in 2024; for 2021 it holds GFS
  short-range forecasts only. July 2021 is replayed with observed rain, like 2005, and lead time
  is measured on three 2024-2025 flood days instead.
- **No municipal waterlogging lists or tide records in the labels yet.** The flood table comes
  from news reports only.
- **pySTEPS and ConvLSTM baselines** were not run in this repository (pySTEPS needs IMERG or
  GSMaP; the ConvLSTM is separate Kaggle work).
- **The tide is predicted, not observed.** Open-Meteo sea level starts in 2023, so a harmonic fit
  gives the astronomical tide for older dates. Storm surge is missing.

## Known limits

- The bias correction was learned on forecasts about a day ahead and is reused unchanged for
  forecasts 2-3 days ahead in the replays.
- IMD is a 0.25 degree grid: neighbouring wards get nearly the same rain, so ward differences
  can only come from the static features, which the model does not yet use.
- Static features have gaps: four wards have no mapped waterway, and the DEM includes buildings.
- The IMD publication time (12:00 IST) used in replays is an assumption.
- Open-Meteo rate-limits bursts of requests; the replay download waits and retries.

## First tasks for Semester 2

1. Review every label row (open the source, set `reviewed = yes`) and add BMC waterlogging
   lists, especially heavy-rain days on which a ward stayed dry.
2. Retrain the flood model and repeat the season count; adopt it only when it warns on fewer
   days than the rule for the same reports caught.
3. Request the IMERG, GSMaP, ERA5 and MOSDAC accounts: ERA5 is needed for Stage 9 (heat) and
   IMERG or GSMaP for the pySTEPS baseline.
4. Then Stage 9 (heat stress) per the plan.

## Reproduce

```bash
git clone <repo> && cd hazardlens
bash start.sh                     # Python env, PostGIS + Redis, wards, map build
source .venv/Scripts/activate     # .venv/bin/activate on Linux/macOS
python -m scripts.demo --serve    # every stage, checks, then the map at http://localhost:8000
```
