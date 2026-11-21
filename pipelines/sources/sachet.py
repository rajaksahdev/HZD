"""Official alerts from SACHET, NDMA's Common Alerting Protocol platform (Stage 14). No account.

SACHET publishes the alerts of IMD, the state disaster authorities, CWC, INCOIS and others as
CAP 1.2 files, listed in RSS feeds that keep only the newest items (Maharashtra: 10, all
India: 100). So history exists only from the first poll on: run this every 15 minutes.

Each poll reads the Maharashtra feed and the all-India feed. Every Maharashtra item is kept;
from the all-India feed only national agencies (INCOIS, CWC, NDMA) and items that name Mumbai or
its neighbours, since those can cover the coast without a state item. For each new item the CAP
file and, when it names one, the polygon file are saved to the raw zone and parsed into
official_alerts. Wards covered: those with at least 5% of their area inside the alert's
polygons; without polygons, the LGD district codes (482 Mumbai, 483 Mumbai Suburban, inferred
from SACHET's alphabetical numbering of Maharashtra's districts).

An Update or Cancel names the messages it replaces in <references>; those are marked
superseded. Nothing official is ever edited or re-signed here.

python -m pipelines.sources.sachet            poll the feeds once
python -m pipelines.sources.sachet --list     the official alerts stored, newest first
"""

import argparse
import re
import sys
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from api.db import OfficialAlert, OfficialAlertWard, OfficialFeedFetch, SessionLocal, init_db
from pipelines import store

NAME = "sachet"
BASE = "https://sachet.ndma.gov.in/cap_public_website"
FEEDS = {
    "maharashtra": f"{BASE}/rss/rss_maharashtra.xml",
    "india": f"{BASE}/rss/rss_india.xml",
}
CAP_URL = f"{BASE}/FetchXMLFile"
NS = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
NATIONAL = ("INCOIS", "CWC", "NDMA")
NEARBY = re.compile(r"mumbai|thane|palghar|raigad|konkan|maharashtra|मुंबई|ठाणे|पालघर|रायगड", re.I)
MIN_SHARE = 0.05  # a ward counts as covered from this share of its area
CITY = ["A", "B", "C", "D", "E", "F-N", "F-S", "G-N", "G-S"]  # Mumbai City district
SUBURBAN = [  # Mumbai Suburban district
    "H-E", "H-W", "K-E", "K-W", "L", "M-E", "M-W", "N", "P-N", "P-S", "R-C", "R-N", "R-S", "S",
    "T",
]  # fmt: skip
LGD_WARDS = {"482": CITY, "483": SUBURBAN}
# CAP severity as a colour on the map. A display convention, not an equivalence: IMD's own
# colour code is not carried in the CAP files.
COLOUR = {"Extreme": "red", "Severe": "orange", "Moderate": "yellow", "Minor": "green"}


# --- Parsing (pure) -------------------------------------------------------------------------


def parse_feed(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    items = []
    for item in root.iter("item"):
        author = item.findtext("author") or ""
        agency = re.search(r"\(([^)]+)\)", author)
        pub = item.findtext("pubDate")
        items.append(
            {
                "guid": (item.findtext("guid") or "").strip(),
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "agency": agency.group(1).strip() if agency else None,
                "published": parsedate_to_datetime(pub) if pub else None,
            }
        )
    return [i for i in items if i["guid"]]


def relevant(item: dict, feed: str) -> bool:
    if feed == "maharashtra":
        return True
    agency = (item["agency"] or "").upper()
    return any(n in agency for n in NATIONAL) or bool(NEARBY.search(item["title"]))


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.strip()) if value and value.strip() else None


def _text(el: ET.Element | None, tag: str) -> str | None:
    value = el.findtext(f"cap:{tag}", namespaces=NS) if el is not None else None
    return value.strip() or None if value else None


def parse_cap(xml: bytes) -> dict:
    """The fields shown on the dashboard. English info block first; the first other language
    (Marathi or Hindi) gives headline_local."""
    root = ET.fromstring(xml)
    infos = root.findall("cap:info", NS)
    english = [i for i in infos if (_text(i, "language") or "").lower().startswith("en")]
    local = [i for i in infos if i not in english]
    main = (english or infos or [None])[0]
    refs = []
    for token in (_text(root, "references") or "").split():
        parts = token.split(",")
        if len(parts) >= 2:
            refs.append(parts[1])
    lgd, polygons, circles, polygon_url, areas = [], [], [], None, []
    for info in infos:
        for p in info.findall("cap:parameter", NS):
            if (_text(p, "valueName") or "").lower() == "polygon url":
                polygon_url = _text(p, "value")
        for area in info.findall("cap:area", NS):
            if info is main:
                areas.append(_text(area, "areaDesc") or "")
            for g in area.findall("cap:geocode", NS):
                if "lgd" in (_text(g, "valueName") or "").lower() and _text(g, "value"):
                    lgd.append(_text(g, "value"))
            polygons += [parse_ring(p.text) for p in area.findall("cap:polygon", NS) if p.text]
            circles += [c.text.strip() for c in area.findall("cap:circle", NS) if c.text]
    return {
        "identifier": _text(root, "identifier"),
        "sender": _text(root, "sender"),
        "sent": _time(_text(root, "sent")),
        "status": _text(root, "status") or "Actual",
        "msg_type": _text(root, "msgType") or "Alert",
        "refs": refs,
        "event": _text(main, "event"),
        "category": _text(main, "category"),
        "severity": _text(main, "severity"),
        "urgency": _text(main, "urgency"),
        "certainty": _text(main, "certainty"),
        "effective": _time(_text(main, "effective")),
        "onset": _time(_text(main, "onset")),
        "expires": _time(_text(main, "expires")),
        "headline": _text(main, "headline"),
        "headline_local": _text(local[0], "headline") if local else None,
        "language_local": _text(local[0], "language") if local else None,
        "instruction": _text(main, "instruction"),
        "area_desc": "; ".join(a for a in areas if a) or None,
        "lgd_codes": sorted(set(lgd)),
        "polygons": [p for p in polygons if len(p) >= 3],
        "circles": circles,
        "polygon_url": polygon_url,
    }


def parse_ring(value: str) -> list[tuple[float, float]]:
    """CAP 'lat,lon lat,lon ...' as (lon, lat) points."""
    points = []
    for pair in value.split():
        lat, lon = pair.split(",")[:2]
        points.append((float(lon), float(lat)))
    return points


def parse_polygon_file(xml: bytes) -> list[list[tuple[float, float]]]:
    """SACHET's polygon file: <alert><identifier/><polygon>lat,lon ...</polygon>...</alert>."""
    root = ET.fromstring(xml)
    rings = [parse_ring(p.text) for p in root.iter() if p.tag.endswith("polygon") and p.text]
    return [r for r in rings if len(r) >= 3]


def multipolygon_wkt(rings: list[list[tuple[float, float]]]) -> str | None:
    if not rings:
        return None
    parts = []
    for ring in rings:
        closed = ring if ring[0] == ring[-1] else [*ring, ring[0]]
        parts.append("((" + ", ".join(f"{x} {y}" for x, y in closed) + "))")
    return "MULTIPOLYGON(" + ", ".join(parts) + ")"


# --- Storing --------------------------------------------------------------------------------


GEOMETRY = """
    WITH parts AS (
      SELECT ST_CollectionExtract(ST_MakeValid(ST_GeomFromText(:wkt, 4326)), 3) AS g
      UNION ALL
      SELECT ST_Buffer(ST_SetSRID(ST_Point(c.lon, c.lat), 4326)::geography, c.km * 1000)::geometry
      FROM unnest(CAST(:lats AS float8[]), CAST(:lons AS float8[]), CAST(:kms AS float8[]))
           AS c(lat, lon, km)
    )
    SELECT ST_Multi(ST_Union(g)) FROM parts WHERE g IS NOT NULL AND NOT ST_IsEmpty(g)
"""


def store_alert(session: Session, cap: dict, item: dict, feed: str, rings, now: datetime) -> str:
    """Insert one CAP message, its wards, and the superseded links both ways."""
    circles = []
    for c in cap["circles"]:
        try:
            centre, km = c.split()
            lat, lon = centre.split(",")
            circles.append((float(lat), float(lon), float(km)))
        except ValueError:
            continue
    rings = rings + cap["polygons"]
    alert = OfficialAlert(
        identifier=cap["identifier"],
        guid=item["guid"],
        feed=feed,
        agency=item.get("agency"),
        sender=cap["sender"],
        sent=cap["sent"] or item.get("published") or now,
        status=cap["status"],
        msg_type=cap["msg_type"],
        refs=" ".join(cap["refs"]) or None,
        event=cap["event"],
        category=cap["category"],
        severity=cap["severity"],
        urgency=cap["urgency"],
        certainty=cap["certainty"],
        effective=cap["effective"],
        onset=cap["onset"],
        expires=cap["expires"],
        headline=cap["headline"],
        headline_local=cap["headline_local"],
        language_local=cap["language_local"],
        instruction=cap["instruction"],
        area_desc=cap["area_desc"],
        lgd_codes=",".join(cap["lgd_codes"])[:400] or None,
        link=item.get("link"),
        match="none",
        fetched_at=now,
    )
    session.add(alert)
    session.flush()
    has_shape = False
    if rings or circles:
        has_shape = session.execute(
            text(
                f"UPDATE official_alerts SET geometry = ({GEOMETRY}) WHERE identifier = :i "
                "RETURNING geometry IS NOT NULL"
            ),
            {
                "i": alert.identifier,
                "wkt": multipolygon_wkt(rings) or "MULTIPOLYGON EMPTY",
                "lats": [c[0] for c in circles],
                "lons": [c[1] for c in circles],
                "kms": [c[2] for c in circles],
            },
        ).scalar()
    wards = []
    if has_shape:
        wards = session.execute(
            text(
                "SELECT w.ward_id, ST_Area(ST_Intersection(w.geometry, a.geometry)) "
                "/ ST_Area(w.geometry) AS share FROM wards w JOIN official_alerts a "
                "ON ST_Intersects(w.geometry, a.geometry) WHERE a.identifier = :i"
            ),
            {"i": alert.identifier},
        ).all()
        wards = [(w, s) for w, s in wards if s >= MIN_SHARE]
        alert.match = "polygon"
    else:
        named = [w for code in cap["lgd_codes"] for w in LGD_WARDS.get(code, [])]
        known = set(session.execute(text("SELECT ward_id FROM wards")).scalars())
        wards = [(w, None) for w in named if w in known]
        alert.match = "lgd" if cap["lgd_codes"] else "none"
    for ward_id, share in wards:
        session.add(OfficialAlertWard(identifier=alert.identifier, ward_id=ward_id, share=share))
    if cap["refs"]:
        session.execute(
            text(
                "UPDATE official_alerts SET superseded_by = :i "
                "WHERE identifier = ANY(:refs) AND superseded_by IS NULL"
            ),
            {"i": alert.identifier, "refs": cap["refs"]},
        )
    later = session.execute(  # a newer message stored before this one
        text(
            "SELECT identifier FROM official_alerts WHERE :i = ANY(string_to_array(refs, ' ')) "
            "ORDER BY sent LIMIT 1"
        ),
        {"i": alert.identifier},
    ).scalar()
    if later:
        alert.superseded_by = later
    return alert.identifier


# --- Polling --------------------------------------------------------------------------------


def _save(day, name: str, content: bytes, url: str) -> None:
    try:
        store.write_raw(NAME, day, name, content, url)
    except FileExistsError:
        pass  # the same item fetched again after the database was reset


def poll(now: datetime | None = None) -> dict:
    """Read both feeds once and store every new relevant CAP message."""
    init_db()
    now = now or datetime.now(UTC)
    day = now.date()
    summary = {}
    for feed, url in FEEDS.items():
        new, items, error = 0, [], None
        try:
            resp = store.http_get(url, timeout=60)
            items = parse_feed(resp.content)
            latest = store.latest_raw(NAME, day, f"feed_{feed}_*.xml")
            if latest is None or latest.read_bytes() != resp.content:
                _save(day, f"feed_{feed}_{now:%Y%m%dT%H%M%SZ}.xml", resp.content, url)
            with SessionLocal() as session:
                seen = set(session.scalars(select(OfficialAlert.guid)))
            for item in items:
                if item["guid"] in seen or not relevant(item, feed):
                    continue
                cap_resp = store.http_get(CAP_URL, params={"identifier": item["guid"]})
                cap = parse_cap(cap_resp.content)
                if not cap["identifier"]:
                    continue
                _save(day, f"cap_{item['guid']}.xml", cap_resp.content, cap_resp.url)
                rings = []
                if cap["polygon_url"]:
                    poly = store.http_get(cap["polygon_url"])
                    _save(day, f"polygon_{item['guid']}.xml", poly.content, poly.url)
                    rings = parse_polygon_file(poly.content)
                with SessionLocal.begin() as session:
                    if session.get(OfficialAlert, cap["identifier"]) is None:
                        store_alert(session, cap, item, feed, rings, now)
                        new += 1
                seen.add(item["guid"])
        except Exception as exc:  # one bad feed must not stop the other
            error = f"{type(exc).__name__}: {exc}"[:300]
        with SessionLocal.begin() as session:
            session.add(
                OfficialFeedFetch(
                    feed=feed, fetched_at=now, ok=error is None, items=len(items), new=new,
                    error=error,
                )
            )  # fmt: skip
        summary[feed] = {"items": len(items), "new": new, "error": error}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official alerts from SACHET (NDMA).")
    parser.add_argument("--list", action="store_true", help="list stored alerts, newest first")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.list:
        init_db()
        with SessionLocal() as session:
            rows = session.execute(
                text(
                    "SELECT a.sent, a.agency, a.msg_type, a.severity, a.event, a.superseded_by, "
                    "string_agg(w.ward_id, ',' ORDER BY w.ward_id) AS wards "
                    "FROM official_alerts a LEFT JOIN official_alert_wards w USING (identifier) "
                    "GROUP BY a.identifier ORDER BY a.sent DESC LIMIT :n"
                ),
                {"n": args.limit},
            ).all()
        for r in rows:
            gone = " (superseded)" if r.superseded_by else ""
            print(
                f"{r.sent:%Y-%m-%d %H:%M}  {r.agency or '?':<18} {r.msg_type:<6} "
                f"{r.severity or '?':<9} {r.event or '?'}  wards: {r.wards or '-'}{gone}"
            )
        return 0
    summary = poll()
    for feed, s in summary.items():
        state = f"error {s['error']}" if s["error"] else "ok"
        print(f"{feed}: {s['items']} items, {s['new']} new official alerts stored, {state}")
    return 1 if all(s["error"] for s in summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
