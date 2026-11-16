"""One module per data source. Each exposes:

fetch(day, force=False) -> list[Path]   download into the raw zone (never edits old files)
harmonize(day) -> xr.Dataset | None     read raw, return data on the pilot grid in UTC
"""

from pipelines.sources import chirps, imd, openmeteo, tide

SOURCES = {
    "openmeteo": openmeteo,
    "tide": tide,
    "chirps": chirps,
    "imd": imd,
}
