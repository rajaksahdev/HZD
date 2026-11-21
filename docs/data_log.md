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
| OpenStreetMap | Roads, drains, shelters | Overpass API; OSMnx for the drive network | 2026-11-16 (waterways; main roads, railways and rivers for MMR and Raigad from 2026-11-19; hospitals, schools, community centres, assembly points and the drive network for Stage 12 from 2026-11-20) | ODbL | © OpenStreetMap contributors | `data/raw/osm/` |
| Copernicus DEM GLO-30 | Terrain (surface heights, 30 m) | Direct download (AWS open data) | 2026-11-16 | confirm | confirm | `data/raw/dem/` |
| ESA WorldCover 2021 v200 | Land cover, 10 m | Direct download (AWS open data) | 2026-11-16 | CC BY 4.0 | confirm | `data/raw/worldcover/` |
| News reports | Flood labels | Web pages, links in `labels/flood_events.csv` | 2026-11-16 | Facts only; no text copied | Link per row | `labels/` |
| OpenStreetMap tiles | Map background | tile.openstreetmap.org, loaded by the browser | 2026-11-19 | OSM tile usage policy: light use only, attribution shown | © OpenStreetMap contributors | not stored |
| MapLibre demo fonts | Ward labels on the map | demotiles.maplibre.org, loaded by the browser | 2026-11-19 | confirm | confirm | not stored |
| Open-Meteo Previous Runs | Archived forecasts made 1-5 days ahead, for replays | Open API, no key | 2026-11-19 | Free for non-commercial use | confirm | `data/raw/previous_runs/` |
| ERA5 hourly reanalysis (via Open-Meteo Historical Weather API) | Heat: thresholds and the forecast correction (Stage 9) | Open API, no key; `models=era5` | 2026-11-19 | ERA5: Copernicus licence (free, attribution); Open-Meteo: CC BY 4.0, free for non-commercial use | Hersbach et al. (2023), ERA5 hourly data on single levels, Copernicus C3S CDS, doi:10.24381/cds.adbb2d47; Open-Meteo.com | `data/raw/era5/` |
| NASA Global Landslide Catalog (export) | Landslide inventory, MMR and Raigad (Stage 10) | data.nasa.gov legacy CSV export, no key | 2026-11-19 | NASA open data; agree to the GLC terms on the NASA landslide pages | Kirschbaum et al. (2010), Natural Hazards 52; Kirschbaum, Stanley and Zhou (2015), Geomorphology 249 | `data/raw/glc/` |
| GSI landslide inventory (Bhukosh) | Landslide inventory (Stage 10), not yet obtained | bhukosh.gsi.gov.in, login; did not respond from outside the portal | - | confirm | Geological Survey of India | `data/raw/gsi/` (placed by hand) |
| Open-Meteo Historical Forecast API (GFS, JMA, UKMO, ECMWF, ICON, GEM) | Archived day-ahead rain from six models, June-September 2021-2025, for the blend and trust levels (Stage 11) | Open API, no key | 2026-11-20 | Free for non-commercial use; CC BY 4.0 | Open-Meteo.com; the national weather services that run each model | `data/raw/blend/` |
| Meta / CIESIN High Resolution Settlement Layer (HRSL), general population | Population per 30 m cell, for exposure, vulnerability and route origins (Stage 12) | Cloud-optimised GeoTIFFs on AWS open data (`dataforgood-fb-data`), read as a window, no key | 2026-11-20 | CC BY 4.0 | Facebook Connectivity Lab and CIESIN, Columbia University (2016), High Resolution Settlement Layer; Data for Good at Meta | `data/raw/population/` |
| BMC temporary shelter list | Designated flood shelters (Stage 12), not yet obtained | Not published as open data | - | confirm | Brihanmumbai Municipal Corporation, Disaster Management Cell | `data/raw/shelters/` (placed by hand) |
