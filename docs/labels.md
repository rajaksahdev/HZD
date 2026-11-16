# Flood labels

`labels/flood_events.csv` is the label table for the flood model. One row is one reported
flooding of a ward on a date, or one ward-day on which flooding is ruled out.

## Current state

Run `python -m pipelines.labels` for the live numbers. On 2026-11-16 the table held 67 rows
covering 59 ward-days: 37 flooded and 22 not flooded, across 8 flood dates between 2005 and 2025
and 20 of the 24 wards.

**No row has been reviewed by a person yet.** The flooded rows were collected by an AI assistant
from news pages. Each row has `reviewed = no` until a team member opens the link, confirms the
place and date, and changes it to `yes`.

## How rows were made

- **Flooded rows (`evidence = news`).** A news report names a place as waterlogged on that date.
  The place is turned into an approximate point, and the ward is the ward polygon that contains
  the point.
  - `confidence = medium`: the page was opened and names a specific spot.
  - `confidence = low`: the row comes from a search summary without opening the page, or the
    report names only an area ("Kurla", "Dadar"), so the point and ward are a guess within it.
- **Not-flooded rows (`evidence = dry_day`).** These are not observations. They are monsoon days
  on which the IMD rainfall grid shows under 2.5 mm over the whole pilot area on that day, the
  day before and the day after. For each such day, the wards flooded in that year's event get a
  not-flooded row.

## Limits

- **Easy negatives.** Dry days are trivially not flooded. The table has no heavy-rain day on
  which a ward is known to have stayed dry, which is what the model most needs. The BMC
  waterlogging lists are the best source for those.
- **Reporting bias.** A ward missing from a flood date is unlabelled, not dry. News names the
  famous spots (Hindmata, Andheri subway, King's Circle) far more often than others.
- **Ward boundaries.** Places on a ward boundary (Dadar, Mumbai Central, the railway subways) can
  fall on either side depending on the point chosen.
- **Dates.** The date is the local calendar date of the report. Flooding after overnight rain is
  dated to the morning it was reported.
- **No depth or duration.** A row says flooding was reported, not how deep or how long.

## Adding rows

1. Add a line to `labels/flood_events.csv` with the date, ward, place, source link and a point.
2. Run `python -m pipelines.labels`. It rejects unknown wards, bad dates, flooded rows without a
   link, and points that fall outside the ward named.
3. Set `reviewed` to `yes` only after opening the source yourself.

## Static features

`python -m pipelines.static` writes one row per ward. Known limits:

- **Slope** comes from Copernicus DEM, a surface model. In built-up wards the heights include
  buildings, so slope is higher and noisier than the ground slope.
- **Drain distance** uses only waterways mapped in OpenStreetMap. Wards B, C, E and F-S have no
  mapped waterway at all, so their large distances reflect missing map data, not missing drains.
  Mumbai's storm drains are mostly underground and unmapped.
- **Ward areas** come from the DataMeet polygons, which include creeks and mangroves in some
  wards.
