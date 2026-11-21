# Vulnerability, impact and routes (Stage 12)

Built 20 Nov 2026 with `python -m models.impact --build --report docs/impact.md --example-run flood-observed-20250819-20261119T093113Z`. Method in `models/impact.py`.

## Inputs

- **Population**: Meta/CIESIN HRSL, about 30 m cells. 114,555 populated cells inside the 24 wards, **13,216,503 people** in all. For comparison, Census 2011 counted 12,442,373 in Greater Mumbai: HRSL spreads census counts over buildings and is an estimate of where people live, not a new count.
- **Hospitals**: 979 OSM amenity=hospital (60 tagged emergency=yes), 847 on the road network. Many are small nursing homes. Left out: 112 named as specialist clinics or labs (eye, dental, diagnostic and similar) and 83 facilities of all kinds with no name, since a route cannot point anyone to them.
- **Shelter candidates**: 638 (580 schools, 55 community centres, 2 shelters, 1 assembly point), 568 on the road network. Not an official list: BMC opens municipal schools as temporary shelters, but the list is not open data.
- **Roads**: OSM drive network, 32,909 junctions and 45,224 segments (3,970 km). 5% of segments have their midpoint below 5 m; 642 are within 250 m of one of 16 reported waterlogging spots.

## Vulnerability per ward

Index = mean of four ranks among the 24 wards (0 lowest, 1 highest): more people per km2; more people on low-lying land; further from a shelter candidate; further from a hospital. It orders wards and does not measure absolute need.

| Ward | People | Per km² | Low-lying share | To shelter (km) | > 1 km from shelter | To hospital (km) | > 2 km from hospital | Index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M-W | 643,442 | 36,972 | 7% | 1.60 | 54% | 1.28 | 24% | 0.77 |
| F-S | 370,687 | 37,872 | 8% | 1.28 | 40% | 0.95 | 10% | 0.73 |
| P-N | 811,506 | 17,371 | 8% | 1.85 | 60% | 1.56 | 21% | 0.71 |
| K-W | 797,248 | 32,471 | 7% | 1.23 | 26% | 1.13 | 6% | 0.67 |
| H-E | 479,845 | 38,637 | 14% | 0.94 | 44% | 0.83 | 0% | 0.66 |
| M-E | 809,539 | 24,470 | 6% | 1.36 | 45% | 1.49 | 29% | 0.65 |
| F-N | 433,337 | 35,278 | 10% | 0.85 | 37% | 0.93 | 3% | 0.63 |
| G-S | 401,223 | 43,171 | 14% | 0.62 | 15% | 0.83 | 0% | 0.61 |
| L | 927,085 | 59,120 | 3% | 0.98 | 44% | 0.68 | 0% | 0.53 |
| A | 344,514 | 30,735 | 8% | 0.68 | 19% | 0.90 | 3% | 0.51 |
| E | 401,117 | 55,145 | 6% | 0.74 | 25% | 0.62 | 0% | 0.50 |
| B | 110,759 | 41,667 | 9% | 0.81 | 29% | 0.56 | 0% | 0.49 |
| P-S | 610,733 | 24,244 | 3% | 1.34 | 50% | 0.99 | 12% | 0.49 |
| R-C | 691,225 | 14,392 | 4% | 1.30 | 43% | 0.91 | 10% | 0.48 |
| K-E | 1,064,985 | 44,441 | 0% | 1.01 | 47% | 0.78 | 2% | 0.45 |
| R-N | 422,172 | 29,775 | 4% | 1.17 | 57% | 0.65 | 0% | 0.40 |
| S | 892,809 | 30,008 | 3% | 0.88 | 32% | 0.80 | 6% | 0.38 |
| N | 564,221 | 21,963 | 5% | 1.06 | 50% | 0.65 | 1% | 0.38 |
| D | 415,693 | 50,550 | 4% | 0.51 | 9% | 0.57 | 0% | 0.38 |
| T | 528,273 | 12,321 | 6% | 0.76 | 19% | 0.75 | 5% | 0.34 |
| C | 101,503 | 53,059 | 4% | 0.51 | 3% | 0.54 | 0% | 0.33 |
| G-N | 427,885 | 48,822 | 6% | 0.48 | 6% | 0.46 | 0% | 0.33 |
| R-S | 542,134 | 29,604 | 1% | 1.28 | 58% | 0.59 | 1% | 0.31 |
| H-W | 424,568 | 47,022 | 3% | 0.50 | 5% | 0.63 | 0% | 0.30 |

Distances are along roads, weighted by where people live. Low-lying share uses the Copernicus surface model, whose heights include buildings, so it undercounts low ground in dense areas.

## Routes in a flood run: `flood-observed-20250819-20261119T093113Z`

Segments avoided: in orange or red wards, midpoint below 5 m or within 250 m of a waterlogging spot. "No safe route" means every way out crosses an avoided segment; the map then shows the usual route in red. People cut off: people whose nearest junction cannot reach any shelter candidate without an avoided segment.

Every ward is red in this run (the Stage 5 flood model over-warns, docs/flood_model.md), so this is the worst case: low-lying roads are avoided across the whole city.

| Ward | Level | Road avoided (km) | Nearest shelter candidate | km (usual) | Nearest hospital | km (usual) | People cut off from a shelter |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| A | red | 13.2 | Holy Name High School | 0.84 | Indian Cancer Society | 0.17 | 16,704 |
| B | red | 4.5 | Orchids The International School, Masjid Bunder | 1.12 (1.00) | Smt.Laxmiben Dhanji Devshi Health Centre | 0.88 (0.60) | 10,427 |
| C | red | 2.5 | The Goan Institute | 0.72 | Yogeshwar Surgical Clinic and Nursing Home - Mumbai | 0.85 (0.81) | 3,937 |
| D | red | 21.2 | Fellowship School | 0.29 (0.13) | Mangeshikar Access Gynecology and Infertility Center | 0.48 | 36,707 |
| E | red | 4.6 | Regina Pacis Convent High School | 0.26 | Rajshree Maternity | 0.14 | 10,321 |
| F-N | red | 39.4 | I.H.Bhatia English Primary School | 0.88 | Sanjeevni hospital | 0.73 | 99,172 |
| F-S | red | 17.1 | JBCN International School | 0.66 | Nowrosjee Wadia Maternity Hospital | 1.40 | 18,513 |
| G-N | red | 10.2 | Victoria High school | 0.55 | smriti narvekar | 0.55 | 29,247 |
| G-S | red | 19.5 | Ganpatrao Kadam Marg BMC School | no safe route (usual 0.75) | ESIS Hospital WORLI | no safe route (usual 1.23) | 35,643 |
| H-E | red | 28.8 | St. Charles Girls' High chool, ground | 1.97 (1.25) | Kalina Hospital | 1.68 (1.18) | 62,990 |
| H-W | red | 7.4 | St Theresa's School | 0.41 | Kohinoor Hospital | 0.48 | 11,803 |
| K-E | red | 4.8 | Holy Family High School | 0.77 | Umang Maternity and Surgical Nursing Home | 1.20 | 30,735 |
| K-W | red | 18.5 | Cosmopolitan Education Society | 0.09 | Siddhivinayak Surgical Centre | 0.44 | 69,417 |
| L | red | 10.9 | St. Judes High school | 0.77 | Uma Hospital | 0.99 | 42,539 |
| M-E | red | 17.4 | Shakuntalam | 0.69 | Apollo Spectra Hospital | 1.38 | 68,833 |
| M-W | red | 11.1 | St. Sebastian's High School | 0.99 | Universal Eye Clinic and Surgical Centre | 0.35 | 46,461 |
| N | red | 9.7 | svdd | 1.48 | Zynova Heart Hospital | 0.29 | 16,595 |
| P-N | red | 22.2 | Municipal Corporation Liberty Garden School | 0.87 | Saraswati Hospital, Mumbai | 0.73 | 166,298 |
| P-S | red | 6.5 | Nandadeep Vidyalay | 0.74 | Trauma Hospital | 0.44 | 8,531 |
| R-C | red | 8.6 | Don Bosco High School | 0.89 | Sailee Hospital and Diagnostic Center | 0.69 | 20,315 |
| R-N | red | 6.0 | Vidhyamandir School | 1.02 | Navneet Hi-Tech Hospital | 0.79 | 20,332 |
| R-S | red | 2.7 | bmc ground | 2.49 | Dhiraj Hospital | 0.60 | 4,558 |
| S | red | 19.9 | St Xavier's School | 0.17 | Naval Hospital | 0.73 | 23,712 |
| T | red | 8.6 | Sheth Karamshi Kanji English School | 0.29 | Hira-Mongi Navneet Hospital | 0.19 | 15,848 |

Run flood-observed-20250819-20261119T093113Z: 315.4 km of road avoided.
- shelter: median 0.77 km by road; longer than usual from 3 wards; wards with no safe route: G-S; 869,638 people cannot reach one.
- hospital: median 0.69 km by road; longer than usual from 3 wards; wards with no safe route: G-S; 842,525 people cannot reach one.

## Limits

- Shelter candidates are places that could serve as shelters, not designated ones; capacity is unknown. Hospital tags do not say which take emergencies.
- Roads are avoided by ward level and terrain, not by observed water on the road.
- The vulnerability index has no social indicators (income, age, housing, slums): no open per-ward data were found.
- Routes start at one point per ward. Footpaths are not in the network.
