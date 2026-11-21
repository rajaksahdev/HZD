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
No account is needed for the demo; the empty account fields in `.env` are for later stages.

## Run the demo

```bash
python -m scripts.demo --serve     # build everything missing, check it, open http://localhost:8000
```

`scripts/demo.py` runs every stage below in order: static ward features, IMD rain for each label
year, Open-Meteo forecasts for 2024-2025, the tide and rain-correction models, the flood model,
today's flood and heat runs, hindcasts of 19 August 2025 (flood) and 23 October 2025 (heat),
the Stage 7 replays, the Stage 10 landslide model, the Stage 11 blend and trust levels and the
Stage 12 population, shelters, hospitals and routes, and the Stage 13 alert engine on the
replayed 19 Aug 2025 runs, and one poll of the official SACHET feed. It then
checks that the
API answers for all 24 wards and that the map is built. Steps already built are skipped, so an
interrupted run resumes; `--force` rebuilds all, `--skip-replay` leaves out Stage 7. The first
run downloads about 265 MB and takes about 30 minutes; if a download fails, run it again. Reports from a demo run go to `data/demo/`.
Results and limits of the semester: `docs/report_sem1.md`.

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
| Open-Meteo | `openmeteo` | Hourly forecast from 6 models: rain, temperature, humidity, wind | none |
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
  from 0.5, red from 0.75 (Stage 13 sends alerts from orange).
- **API** (`api/main.py`, docs at `/docs`): `GET /v1/risk?lat=&lon=`, `GET /v1/wards/{id}/risk`,
  `GET /v1/wards` (GeoJSON for the map), `GET /v1/runs`, `GET /v1/health`. Without `run_id`
  the newest run valid now is used. Every answer carries the data age and the on-screen
  warnings: TEST data, out of date, stale data (newest input over 3 hours old), wards at
  orange or red, the flood model caveat (when any ward is above green), and the wards at low
  trust (Stage 11).
- **Map** (`web/`, React + MapLibre): wards coloured by severity, a red banner when any ward is
  orange or red, data age (red when stale), a run picker, and a click anywhere for that
  place's ward and risk. `?run=<run_id>` opens a given run. For development run
  `npm run dev` in `web/` (port 5173; set `API_URL` if the API is not on port 8000).

## Replay mode (Stage 7)

```bash
python -m models.replay --fetch 2024-05 2024-10    # archived forecasts, one request per month
python -m models.replay --fetch 2025-05 2025-10    # (Open-Meteo rate-limits: wait if you get 429)
python -m models.replay --report docs/replay.md    # replay 5 flood days, save TEST runs, lead time
```

- **Forecast replays** (8 Jul 2024, 26 May 2025, 19 Aug 2025): each flood day is re-run as if
  issued 0 to 3 days before 00:00 IST, using for every hour only an archived forecast made at
  least that far ahead (Open-Meteo Previous Runs archive, from 2024) and IMD rain only once it
  would have been published.
- **Observed replays** (26 Jul 2005, 18 Jul 2021): no forecasts made days ahead are archived for
  those years, so they use observed IMD rain and measure no lead time.
- Every replay is saved as a TEST run (`replay-YYYYMMDD-leadNd`, `replay-YYYYMMDD-observed`);
  open one on the map with `/?run=<run id>`. Results and limits: `docs/replay.md`.
- **Result:** the model warned all 16 reported ward-days 3 days ahead, but replayed 3 days ahead
  it also warns on 66% of monsoon days. The heavy-rain rule caught 10 of 16 at 3 days while
  warning on 10% of days.

## Heat stress (Stage 9)

```bash
python -m pipelines.run --start 2024-01-31 --end 2025-12-31 --sources openmeteo  # full years
python -m models.heat --fetch 1991 2025          # ERA5 through Open-Meteo, no account needed
python -m models.heat --train 2024-02-01 2024-07-31 --validate 2024-08-01 2024-12-31 \
    --test 2025-01-01 2025-12-31 --report docs/heat.md
python -m models.heat --date today               # heat level per ward -> PostGIS (TEST)
python -m models.heat --date 2025-10-23 --source era5   # a past hot day from ERA5 (TEST)
```

- **Index:** apparent temperature (Steadman, shade), from temperature, humidity and wind; the
  daily maximum over the IST day.
- **Levels:** yellow, orange and red from the 90th, 95th and 99th percentiles of ERA5 1991-2020
  over Mumbai (36.9, 37.8 and 39.4 °C). Stage 13 sends alerts from orange.
- **Forecast:** ECMWF, GFS and ICON, quantile-mapped to ERA5; the way to combine them is chosen
  on August-December 2024 and scored once on 2025. Results and limits: `docs/heat.md`.
- **Map:** the Flood / Heat switch (or `/?hazard=heat`) shows the heat level and the "feels
  like" maximum per ward. The API takes `hazard=heat` on `/v1/wards`, `/v1/risk`,
  `/v1/wards/{id}/risk` and `/v1/runs`; flood stays the default.
- **City scale:** ERA5 has two land cells over Mumbai, so wards differ little. Ward-level heat
  needs land-surface temperature (later stage).

## Landslide susceptibility (Stage 10)

```bash
python -m models.landslide --fetch                       # catalog, DEM, land cover, OSM lines
python -m models.landslide --report docs/landslide.md    # 1 km cells, spatial split, final model
```

- **Area:** a box around the Mumbai Metropolitan Region and Raigad (72.75-73.85 E,
  17.80-19.80 N), 0.01 degree cells, about 18,600 on land.
- **Inventory:** NASA Global Landslide Catalog (news-based, 2007-2017), events located to 1 km
  or better: 25 landslide cells. The GSI inventory needs a Bhukosh login: put its export
  (GeoJSON, GeoPackage or shapefile) in `data/raw/gsi/` and rerun `--report`.
- **Models:** XGBoost and random forest against slope alone, scored on held-out 0.25 degree
  blocks (PR-AUC, and recall in the top 10% / 20% of each held-out area's land).
- **Result:** the final model (terrain, rivers and monsoon rain) puts 14 of 25 held-out
  landslide cells in the top 10% of land. Land cover and road distance score higher but learn
  where landslides get reported, so they are left out. Details and limits:
  `docs/landslide.md`. Saved: `data/processed/models/landslide_xgb.json` and the map
  `data/processed/landslide/susceptibility.tif`. Not on the web map yet.

## Blending and bust detection (Stage 11)

```bash
python -m pipelines.run --start 2022-05-31 --end 2022-09-30 --sources imd   # and 2023
python -m models.blend --fetch 2021 2025           # archived forecasts, June-September, by month
python -m models.blend --report docs/blend.md      # fit, score, reliability plot, saved model
```

- **Archive:** Open-Meteo's stitched short-lead forecasts (about a day ahead) from GFS and JMA
  (2021), UKMO (2022), ECMWF, ICON and GEM (2023), scored against IMD for June-September
  2021-2025.
- **Blend:** each model quantile-mapped to IMD, then weighted by skill. **Probability** of
  moderate and heavy rain from a logistic regression; the reliability plot is
  `docs/blend_reliability.svg`.
- **Trust:** high, medium or low from the spread of the corrected models. **Bust:** forecast and
  IMD two or more rain classes apart. Scored leaving one monsoon out at a time, plus a forward
  check on 2025.
- **Result:** low trust busts more often (4.7% of cell-days against 2.8% at high trust), but the
  ranges overlap and the 2025 forward check shows little difference. For heavy-rain forecasts the
  gap is clearer: 4% bust at high trust, 9% at medium, 17% at low. The blend's rain amount is no
  better than Stage 4's, so flood risk keeps Stage 4's amount. Spread does not grade heat
  forecasts, so heat keeps a fixed trust. Details and limits: `docs/blend.md`.
- **In the runs:** every flood risk run sets each ward's trust from the forecast windows it used
  (observed rain is high trust), and the API names low-trust wards. Live forecasts now fetch all
  six models; Stages 4 and 9 still use ECMWF, GFS and ICON. Saved model:
  `data/processed/models/blend.json`.

## Vulnerability and routes (Stage 12)

```bash
python -m models.impact --fetch                              # population, facilities, roads
python -m models.impact --build --report docs/impact.md      # vulnerability per ward -> PostGIS
python -m models.impact --routes RUN_ID                      # routes avoiding a run's risky roads
```

- **Population:** Meta/CIESIN High Resolution Settlement Layer, about 30 m cells: 13.2 million
  people in the 24 wards (Census 2011: 12.4 million).
- **Facilities:** OpenStreetMap hospitals (specialist clinics and labs left out) and shelter
  candidates: schools, community centres and assembly points. BMC opens municipal schools as
  flood shelters but its list is not open data; put an official list (CSV: name, lat, lon) in
  `data/raw/shelters/` and it replaces the candidates.
- **Roads:** the OSMnx drive network of the wards plus 1 km (33,000 junctions). Road distance
  from every populated cell to the nearest shelter candidate and hospital.
- **Vulnerability index:** mean rank among the 24 wards of density, share of people on low-lying
  land, and road distance to a shelter and to a hospital (0 to 1, relative).
- **Impact:** people at risk = probability x people exposed (flood: people on low-lying land;
  heat: everyone); impact score = people at risk x (0.5 + vulnerability index).
- **Routes:** in a flood run, segments in orange or red wards that are low-lying or within
  250 m of a reported waterlogging spot are avoided. Each ward gets a route from its
  population-weighted centre to the nearest reachable shelter candidate and hospital, or
  "no safe route". `models.risk` computes routes for each new flood run; other runs fall back to
  the baseline routes (nothing avoided) with a warning.
- **API and map:** `GET /v1/impact`, `GET /v1/wards/{id}/impact` (with route GeoJSON),
  `GET /v1/facilities`; `/v1/wards` carries population and people at risk. The map lists people
  at risk per ward, shows facilities when zoomed in, and draws the picked ward's routes (dashed
  red when there is no safe route). Results and limits: `docs/impact.md`.

## Alert engine (Stage 13)

```bash
python -m alerts.keys --new                       # Ed25519 signing key in secrets/ (not committed)
python -m alerts.operators --add NAME             # operator: password + authenticator app (TOTP)
python -m alerts.engine --targets --report docs/alerts.md   # false-alarm share per ward
python -m alerts.engine --run latest --hazard flood         # after each risk run
python -m alerts.engine --pending                 # drafts waiting for an operator
python -m alerts.engine --approve ID --operator NAME        # asks for password and code
python -m alerts.engine --cancel ALERT_ID --operator NAME --reason "false alarm"
python -m alerts.engine --cap ALERT_ID            # CAP 1.2 XML
```

- **Rules:** a ward at orange or red gets an alert (action code per hazard and level), expiring
  at the end of the risk window. The next run sends an update when the level, trust or window
  changes, and a cancel when the ward drops below orange. Orange goes out at once; red, any
  lowering or ending of a red alert, a new low-trust alert and a new alert in a ward over its
  false-alarm target wait for an operator. Landslide susceptibility sends no alerts.
- **Messages:** JSON in the shared `Alert` schema (1.3): `alert_id`, `sequence` (1, 2, ... per
  alert, so receivers drop old or replayed messages), `message_type` (alert, update, cancel),
  expiry, `key_id` and an Ed25519 signature over the canonical JSON. TEST unless the run is live
  and `ALERTS_LIVE=1`. The private key is read only by the engine, never by the API.
- **Operators:** two-factor login (scrypt password and a one-time TOTP code, RFC 6238); a code
  works once; five failures lock the operator for 15 minutes.
- **CAP 1.2:** Alert, Update or Cancel with references to earlier messages, ward polygon and
  geocode, signature as parameters; checked against the OASIS CAP 1.2 schema.
- **False-alarm target:** at most 50% of a ward's flood alerts unconfirmed by a flood report
  (`--set-target` per ward). On the 2024-2025 day-ahead replays all 24 wards are over it, so new
  flood alerts wait for an operator. Results and limits: `docs/alerts.md`.
- **API:** `GET /v1/alerts` (in force now or `?at=`), `GET /v1/alerts/{id}` (all messages),
  `GET /v1/alerts/{id}/cap`, `GET /v1/alert-keys`.

## Official alerts (Stage 14)

```bash
python -m pipelines.sources.sachet            # poll SACHET (NDMA); cron every 15 minutes
python -m pipelines.sources.sachet --list     # official alerts stored, newest first
```

- **Source:** SACHET, NDMA's CAP platform: IMD, Maharashtra SDMA, CWC, INCOIS. The Maharashtra
  feed (all items) and the all-India feed (national agencies and items naming Mumbai or its
  neighbours). Feeds keep only the newest 10 and 100 items, so history starts at the first poll.
- **Wards:** those with 5% or more of their area inside an alert's polygons; without a shape,
  LGD district codes 482 (Mumbai) and 483 (Mumbai Suburban). Updates and cancels replace the
  messages they reference.
- **API:** `GET /v1/official-alerts?at=&ward_id=`: in force for the 24 wards, the newest from
  Maharashtra, and the time of the last poll. `/v1/alerts` also counts drafts waiting.
- **Dashboard:** official and HazardLens alerts side by side, wards under an official alert
  outlined in dashed blue, the latest Maharashtra messages, English and Marathi headlines as
  published. Details and limits: `docs/official.md`.

## Layout

```
api/        config, shared schemas (api/schemas.py), database tables, main.py (FastAPI)
pipelines/  raw store, pilot grid, one fetcher per source (sources/sachet.py: official
            alerts), runner, QC checks, fault detector
scripts/    load_wards.py, demo.py (the Semester 1 demo end to end)
models/     forecast.py (bias correction and skill scores), tide.py (harmonic tide),
            flood.py (XGBoost flood model), risk.py (risk run for every ward -> PostGIS),
            replay.py (past flood days as if live, lead time), heat.py (heat stress),
            landslide.py (landslide susceptibility, MMR and Raigad), blend.py (model blend,
            trust levels, bust detection), impact.py (population, shelters, hospitals,
            vulnerability, people at risk, routes)
alerts/     keys.py (Ed25519), operators.py (two-factor login), engine.py (rules, messages,
            false-alarm targets), cap.py (CAP 1.2 export)
web/        the map (React + MapLibre)
assistant/  later stages
labels/     flood_events.csv, the hand-kept flood label table
docs/       data_log.md (source, date, licence, citation), qc_report.md, labels.md, forecast_skill.md,
            flood_model.md, replay.md, report_sem1.md (Semester 1 report), heat.md,
            landslide.md, blend.md and blend_reliability.svg, impact.md, alerts.md,
            official.md
data/       raw and processed files, not committed
```

## Rules

- Every module exchanges the Pydantic models in `api/schemas.py`, keyed by `ward_id`
  (`A`, `F-N`, `K-W`, ...). Change a schema only by bumping `SCHEMA_VERSION`.
- Split training data by time, never randomly.
- Every demo alert is marked TEST.
- Secrets live only in `.env`.
