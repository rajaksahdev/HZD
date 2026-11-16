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

## Layout

```
api/        config, shared schemas (api/schemas.py), database tables
pipelines/  raw store, pilot grid, one fetcher per source, runner, QC checks, fault detector
scripts/    load_wards.py
models/ assistant/ web/   later stages
docs/       data_log.md (source, date, licence, citation), qc_report.md
data/       raw and processed files, not committed
```

## Rules

- Every module exchanges the Pydantic models in `api/schemas.py`, keyed by `ward_id`
  (`A`, `F-N`, `K-W`, ...). Change a schema only by bumping `SCHEMA_VERSION`.
- Split training data by time, never randomly.
- Every demo alert is marked TEST.
- Secrets live only in `.env`.
