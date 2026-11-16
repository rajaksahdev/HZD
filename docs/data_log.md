# Data log

One row per source. Add the row the day you first download from it. Check licence terms on the
source site before relying on them: the "Licence / terms" column is what we understood on the
access date, and the entries marked "confirm" have not been checked yet.

| Source | Used for | Access | First accessed | Licence / terms | Citation | Stored at |
|---|---|---|---|---|---|---|
| DataMeet BMC wards | 24 ward boundaries | Direct download (GitHub) | 2026-11-16 | CC BY-SA 2.5 IN | DataMeet Trust, Municipal Spatial Data, Mumbai | `data/raw/wards/` |
| NASA GPM IMERG | Rain training and verification | Earthdata account, `earthaccess` | not yet | confirm | confirm | `data/raw/imerg/` |
| JAXA GSMaP_NOW | Real-time rainfall | JAXA registration | not yet | confirm | confirm | `data/raw/gsmap/` |
| CHIRPS v3 | Daily rain history | Direct download | 2026-11-16 | confirm | confirm | `data/raw/chirps/` |
| ERA5 | Temp, humidity, wind, soil moisture | Copernicus CDS, `cdsapi` | not yet | confirm | confirm | `data/raw/era5/` |
| IMD gridded | India rain and temperature grids | `imdlib` | 2026-11-16 | confirm | confirm | `data/raw/imd/` |
| MOSDAC (INSAT) | Low-latency rainfall | Registration, needs approval | not yet | Research use only; must cite | confirm | `data/raw/mosdac/` |
| Open-Meteo | Forecasts and tides | Open API, no key | 2026-11-16 | Free for non-commercial use | confirm | `data/raw/openmeteo/` |
| OpenStreetMap | Roads, drains, shelters | Overpass API | 2026-11-16 (waterways only) | ODbL | © OpenStreetMap contributors | `data/raw/osm/` |
| Copernicus DEM GLO-30 | Terrain (surface heights, 30 m) | Direct download (AWS open data) | 2026-11-16 | confirm | confirm | `data/raw/dem/` |
| ESA WorldCover 2021 v200 | Land cover, 10 m | Direct download (AWS open data) | 2026-11-16 | CC BY 4.0 | confirm | `data/raw/worldcover/` |
| News reports | Flood labels | Web pages, links in `labels/flood_events.csv` | 2026-11-16 | Facts only; no text copied | Link per row | `labels/` |
