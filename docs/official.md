# Official alerts beside HazardLens (Stage 14)

Written 21 Nov 2026 after the first polls of the feed. Code in `pipelines/sources/sachet.py`;
API in `api/main.py` (`GET /v1/official-alerts`); dashboard panel in `web/src/App.jsx`.

## The feed

SACHET (sachet.ndma.gov.in) is NDMA's Common Alerting Protocol platform. IMD's regional
centres, the state disaster authorities, CWC, INCOIS and others publish their alerts there as
CAP 1.2 files. No account is needed. What the first polls showed:

- **RSS feeds keep only the newest items:** 10 for Maharashtra (`rss_maharashtra.xml`), 100 for
  all India (`rss_india.xml`). There is no public archive, so HazardLens keeps every message it
  sees from its first poll on. Poll every 15 minutes:
  `python -m pipelines.sources.sachet`.
- **One CAP file per item** (`FetchXMLFile?identifier=<guid>`), often with two info blocks:
  English (`en-IN`) and Marathi (`MR`). The Marathi block is sometimes first, so the parser picks
  the English one by language code and keeps the Marathi headline as published.
- **Areas** are LGD district codes plus a separate polygon file (`FetchPolygonXMLFile`, one
  polygon per district, `lat,lon` pairs). The stored polygons match district sizes (Solapur:
  14,857 km²).
- **Updates and cancels** name the messages they replace in `<references>`. Many Maharashtra
  items are Updates sent by the state authority (`Maharashtra-SDMA`) on an IMD Mumbai message.
- On 21 Nov 2026 (dry season) the Maharashtra feed held ten IMD Mumbai thunderstorm and rain
  messages from 2 Nov, for districts 3 to 210 km from the pilot wards, and none for Mumbai.

## What HazardLens does with it

- Every Maharashtra item is stored. From the all-India feed only national agencies (INCOIS,
  CWC, NDMA) and items naming Mumbai or its neighbours, since those can cover the coast without
  a state item. Each CAP and polygon file goes to the raw zone (`data/raw/sachet/`) once.
- **Wards covered:** those with at least 5% of their area inside the alert's polygons (or CAP
  circles). Without any shape, the LGD codes: 482 Mumbai (wards A to G/N), 483 Mumbai Suburban
  (H/E to T). These two codes are inferred: SACHET numbers Maharashtra's districts
  alphabetically (Ahmednagar 466, Aurangabad 469, Jalgaon 478, Kolhapur 480, Osmanabad 488,
  Pune 490, Ratnagiri 492, Sindhudurg 495, Solapur 496 all checked against named alerts).
- **In force at a time:** status Actual, sent by then, not expired, not a Cancel, and not
  replaced by a message sent by then. A message without an expiry counts for a day.
- **Never changed:** headlines, times and severities are shown as published, with a link to
  the CAP file. HazardLens does not re-sign or re-send official alerts.

## On the dashboard

The panel's **Alerts** section shows two columns side by side: official alerts in force for
the 24 wards (agency, event, CAP severity, expiry, English and Marathi headline, wards) and
HazardLens alerts in force (Stage 13), with the number of drafts waiting for an operator.
Wards under an official alert get a dashed blue outline on the map, so the official area can
be compared with the HazardLens levels underneath. Below, the five newest official messages
from Maharashtra give context when nothing covers Mumbai, and the time of the last poll shows
how fresh the official list is. A past run shows both lists as they stood when it was issued.

CAP severity is drawn as a colour (Extreme red, Severe orange, Moderate yellow, Minor green)
for display only: IMD's own colour code is not carried in the CAP files, and a CAP "Moderate"
is not the same as a HazardLens yellow.

## Limits

- History starts at the first poll; replays of 2024 and 2025 have no official alerts to
  compare with. A season of polling is needed before HazardLens and official alerts can be
  compared ward by ward (hits, misses, lead time).
- LGD codes 482 and 483 are inferred, not read from a published table; polygons are used
  whenever a message has one.
- The all-India filter may miss a national item that names neither Mumbai nor a nearby
  district; the Maharashtra feed is the main source.
- The feed is polled, not pushed: an alert can appear up to 15 minutes late.
