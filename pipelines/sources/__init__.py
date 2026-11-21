"""One module per data source. Each exposes:

fetch(day, force=False) -> list[Path]   download into the raw zone (never edits old files)
harmonize(day) -> xr.Dataset | None     read raw, return data on the pilot grid in UTC

sachet.py (official CAP alerts, Stage 14) is not gridded and is polled on its own:
python -m pipelines.sources.sachet
"""

from pipelines.sources import chirps, imd, openmeteo, tide

SOURCES = {
    "openmeteo": openmeteo,
    "tide": tide,
    "chirps": chirps,
    "imd": imd,
}
