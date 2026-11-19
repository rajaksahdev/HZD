# HazardLens

Multilingual early-warning and action platform for floods, heat and landslides. Pilot: Mumbai,
24 administrative wards. Decision-support prototype: official warnings remain with IMD, NDMA and
state authorities.

## Setup on a new laptop

Needs Git, Python (any 3.x; the script installs 3.11 itself) and Docker Desktop running.
On Windows use Git Bash or WSL2.

```bash
git clone <repo> && cd hazardlens
bash start.sh          # venv, hooks, PostGIS + Redis, loads the 24 wards
source .venv/Scripts/activate    # .venv/bin/activate on Linux/macOS
pytest
```

Then fill in `.env`. The database listens on host port 5433 (set `DB_PORT` to change it).

## Data pipeline (Stage 1)

```bash
python -m pipelines.run --start 2025-07-18 --end 2025-07-21   # rebuild any date range
python -m pipelines.run --days-back 2 --days-ahead 2          # the scheduled run
python -m pipelines.run --start 2025-07-20 --no-fetch         # rebuild clean data from raw only
```

- Raw zone: `data/raw/{source}/YYYY/MM/DD/`, never edited; every file is logged with its checksum
  in `data/raw/manifest.jsonl`.
- Clean zone: `data/processed/{source}/YYYY/MM/DD.nc`, on the 0.1° pilot grid, time in UTC.

| Source | Key | What | Account |
|---|---|---|---|
| Open-Meteo | `openmeteo` | Hourly forecast from 3 models: rain, temperature, humidity, wind | none |
| Open-Meteo marine | `tide` | Hourly sea level near Mumbai harbour | none |
| CHIRPS v3 | `chirps` | Daily rainfall, 0.05°, about a month behind | none |
| IMD gridded | `imd` | Daily rainfall, 0.25° | none |
| IMERG, GSMaP_NOW, ERA5, MOSDAC | not built yet | | needed |

Schedule (Linux/WSL cron, hourly):

```
5 * * * * cd /path/to/hazardlens && .venv/bin/python -m pipelines.run --days-back 2 --days-ahead 2 >> data/cron.log 2>&1
```

On Windows, create a Task Scheduler task that runs
`.venv\Scripts\python.exe -m pipelines.run --days-back 2 --days-ahead 2` from the project folder.

## Validation and QC (Stage 2)

Every dataset is checked before it is written to the clean zone:

- **Schema:** variables, dimensions, units, pilot grid and time steps. A failure is an error and
  the file is not written.
- **Range:** values outside physical limits for Mumbai are set to missing and counted.
- **Gap:** missing time steps and missing values are reported.

Range and gap findings are appended to `data/processed/qc_log.jsonl`.

```bash
python -m pipelines.qc --start 2025-06-15 --end 2025-08-31     # re-check clean files, list missing days
python -m pipelines.anomaly --start 2025-06-15 --end 2025-08-31 --report docs/qc_report.md
```

The second command trains an Isolation Forest on hourly series and reports how many injected
spikes, flat lines and drifts it catches. Results are in `docs/qc_report.md`. The detector is not
yet applied to live data: there are no station feeds to apply it to.

## Labels and static features (Stage 3)

```bash
python -m pipelines.static                  # terrain, drains and land cover per ward -> CSV and PostGIS
python -m pipelines.labels                  # validate labels/flood_events.csv and print a summary
python -m pipelines.labels --dry-days 2024  # list dry monsoon days in a year
```

- **Static features** (`ward_static` table, `data/processed/static/ward_features.csv`): area,
  elevation, share of land below 5 m, slope, distance to the nearest mapped waterway, waterway
  length per km², and built-up, green, water and wetland shares.
- **Flood labels** (`labels/flood_events.csv`, kept in git): one row per reported flooding of a
  ward on a date, with the source link. Add rows by hand; the validator checks ward ids, dates,
  sources and that each point lies in the ward named. See `docs/labels.md` for the rules and
  limits.

## Forecast core (Stage 4)

```bash
python -m pipelines.run --start 2024-05-30 --end 2024-10-01 --sources openmeteo,imd   # backfill
python -m models.forecast --train 2024-06-01 2024-09-30 --test 2025-06-01 2025-09-30     --report docs/forecast_skill.md
```

Scores daily rain from ECMWF, GFS and ICON, their average, and the average after quantile-mapping
bias correction, against IMD observed rain and a persistence baseline. The skill table (POD, FAR,
CSI) is in `docs/forecast_skill.md`. The learned correction is saved to
`data/processed/models/rain_quantile_map.json`. The pySTEPS baseline is not run: it needs IMERG
or GSMaP rain maps, which need accounts.

## Flood model (Stage 5)

```bash
python -m pipelines.run --start 2005-05-15 --end 2005-10-05 --sources imd   # IMD rain per label year
python -m models.tide --fit 2023-01-01 2024-12-31 --check 2025-01-01 2025-10-31
python -m models.flood --test-from 2024 --report docs/flood_model.md
python -m models.flood --predict 2025-08-19       # hindcast every ward on a past day
```

- **Tide** (`models/tide.py`): Open-Meteo sea level only starts in 2023, so a harmonic fit on
  2023-2024 predicts the astronomical tide for any date. Checked on 2025: daily high tide off
  by about 0.10 m. Storm surge is not included.
- **Flood model** (`models/flood.py`): XGBoost on IMD rain around the day, the day's high tide
  and the Stage 3 ward features, one row per labelled ward-day. Scored on held-out years
  (train before 2024, test 2024-2025) and year by year (each year trained only on earlier
  years), against a heavy-rain rule. It also counts how often each would warn over a whole
  monsoon season, which is the honest false-alarm check while all labelled negatives are dry
  days. Results: `docs/flood_model.md`. Saved model: `data/processed/models/flood_xgb.json`.
- **Current state:** the model warns on about three times as many days as the heavy-rain rule,
  because the labels have no heavy-rain day on which a ward stayed dry. It needs reviewed
  labels and BMC waterlogging lists before it is better than the rule.

## API and map (Stage 6)

```bash
python -m pipelines.run --days-back 4 --days-ahead 1 --sources imd,openmeteo
python -m models.risk --date today --rain forecast         # today's flood risk -> PostGIS
python -m models.risk --date 2025-08-19 --rain observed    # a past flood day (hindcast, TEST)
python -m models.risk --list                                # runs in the database
cd web && npm ci && npm run build && cd ..                  # build the map once (needs Node)
uvicorn api.main:app --port 8000                            # map at http://localhost:8000
```

- **Risk run** (`models/risk.py`): the Stage 5 model for every ward for one local (IST) day,
  saved as a row in `risk_runs` plus one `ward_risk` row per ward. `forecast` mode uses the
  Stage 4 corrected forecast for the day (IMD for earlier days where published, else the
  forecast); `observed` mode is a hindcast from IMD. Every run is marked TEST unless
  `--live` is given in forecast mode. Severity: green below 0.25, yellow from 0.25, orange
  from 0.5, red from 0.75 (provisional until Stage 13).
- **API** (`api/main.py`, docs at `/docs`): `GET /v1/risk?lat=&lon=`, `GET /v1/wards/{id}/risk`,
  `GET /v1/wards` (GeoJSON for the map), `GET /v1/runs`, `GET /v1/health`. Without `run_id`
  the newest run valid now is used. Every answer carries the data age and the on-screen
  warnings: TEST data, out of date, stale data (newest input over 3 hours old), wards at
  orange or red, low trust.
- **Map** (`web/`, React + MapLibre): wards coloured by severity, a red banner when any ward is
  orange or red, data age (red when stale), a run picker, and a click anywhere for that
  place's ward and risk. `?run=<run_id>` opens a given run. For development run
  `npm run dev` in `web/` (port 5173; set `API_URL` if the API is not on port 8000).

## Layout

```
api/        config, shared schemas (api/schemas.py), database tables, main.py (FastAPI)
pipelines/  raw store, pilot grid, one fetcher per source, runner, QC checks, fault detector
scripts/    load_wards.py
models/     forecast.py (bias correction and skill scores), tide.py (harmonic tide),
            flood.py (XGBoost flood model), risk.py (risk run for every ward -> PostGIS)
web/        the map (React + MapLibre)
assistant/  later stages
labels/     flood_events.csv, the hand-kept flood label table
docs/       data_log.md (source, date, licence, citation), qc_report.md, labels.md, forecast_skill.md,
            flood_model.md
data/       raw and processed files, not committed
```

## Rules

- Every module exchanges the Pydantic models in `api/schemas.py`, keyed by `ward_id`
  (`A`, `F-N`, `K-W`, ...). Change a schema only by bumping `SCHEMA_VERSION`.
- Split training data by time, never randomly.
- Every demo alert is marked TEST.
- Secrets live only in `.env`.
