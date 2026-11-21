"""Stage 14: official alerts from SACHET. Offline: feed and CAP parsing. With the database:
ward matching, updates that replace a message, the API, and one poll against a fake feed."""

import secrets
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from pipelines.sources import sachet

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Maharashtra: CAP Disaster Alert Feeds</title>
<item><title>Heavy rain likely over Mumbai</title><category>Met</category>
<link>https://sachet.ndma.gov.in/cap_public_website/FetchXMLFile?identifier=111</link>
<author>controlroom@ndma.gov.in (IMD Mumbai)</author><guid isPermaLink="false">111</guid>
<pubDate>Sat, 21 Nov 2026 05:30:16 GMT</pubDate></item>
<item><title>Rain over Gomati, Tripura</title><author>x (IMD Agartala)</author>
<guid>222</guid></item>
<item><title>High waves along the coast</title><author>x (INCOIS)</author><guid>333</guid></item>
</channel></rss>"""


def cap_xml(identifier: str, sent: str, expires: str, msg_type="Alert", refs="", polygon="",
            lgd=("482",)) -> bytes:  # fmt: skip
    """A SACHET-style message: Marathi block first, then English, LGD geocodes, polygon URL."""
    references = f"<cap:references>{refs}</cap:references>" if refs else ""
    geocodes = "".join(
        f"<cap:geocode><cap:valueName>LGD District Code</cap:valueName><cap:value>{c}</cap:value>"
        "</cap:geocode>"
        for c in lgd
    )
    poly = f"<cap:polygon>{polygon}</cap:polygon>" if polygon else ""
    info = """<cap:info><cap:language>{lang}</cap:language><cap:category>Met</cap:category>
<cap:event>Heavy Rain</cap:event><cap:urgency>Expected</cap:urgency>
<cap:severity>Severe</cap:severity><cap:certainty>Likely</cap:certainty>
<cap:effective>{sent}</cap:effective><cap:onset>{sent}</cap:onset>
<cap:expires>{expires}</cap:expires>
<cap:headline>{headline}</cap:headline>
<cap:instruction>Please follow SDMA guidelines.</cap:instruction>
<cap:parameter><cap:valueName>Polygon URL</cap:valueName>
<cap:value>https://example.invalid/poly?identifier=1</cap:value></cap:parameter>
<cap:area><cap:areaDesc>{area}</cap:areaDesc>{poly}{geocodes}</cap:area></cap:info>"""
    return f"""<cap:alert xmlns:cap="urn:oasis:names:tc:emergency:cap:1.2">
<cap:identifier>{identifier}</cap:identifier><cap:sender>IMD-Mumbai</cap:sender>
<cap:sent>{sent}</cap:sent><cap:status>Actual</cap:status><cap:msgType>{msg_type}</cap:msgType>
<cap:scope>Public</cap:scope>{references}
{info.format(lang="MR", sent=sent, expires=expires, headline="मुंबईत मुसळधार पाऊस", area="मुंबई",
             poly=poly, geocodes=geocodes)}
{info.format(lang="en-IN", sent=sent, expires=expires, headline="Heavy rain over Mumbai",
             area="Mumbai district of Maharashtra", poly=poly, geocodes=geocodes)}
</cap:alert>""".encode()  # fmt: skip


POLYGON_FILE = b"""<alert><identifier>x</identifier>
<polygon>18.90,72.80 18.90,72.86 18.99,72.86 18.99,72.80</polygon>
<polygon>40.0,10.0 40.0,10.1 40.1,10.1</polygon></alert>"""


def test_feed_items_and_which_are_kept():
    items = sachet.parse_feed(FEED)
    assert [i["guid"] for i in items] == ["111", "222", "333"]
    assert items[0]["agency"] == "IMD Mumbai"
    assert items[0]["published"] == datetime(2026, 11, 21, 5, 30, 16, tzinfo=UTC)
    assert all(sachet.relevant(i, "maharashtra") for i in items)
    assert [sachet.relevant(i, "india") for i in items] == [True, False, True]


def test_cap_takes_the_english_block_and_keeps_the_local_headline():
    cap = sachet.parse_cap(
        cap_xml("IN-1_29", "2026-07-20T10:00:00+05:30", "2026-07-20T13:00:00+05:30",
                msg_type="Update", refs="IMD-Mumbai,IN-0_46,2026-07-20T09:00:00+05:30 S,IN-9,t")
    )  # fmt: skip
    assert cap["identifier"] == "IN-1_29" and cap["msg_type"] == "Update"
    assert cap["headline"] == "Heavy rain over Mumbai"
    assert cap["headline_local"] == "मुंबईत मुसळधार पाऊस" and cap["language_local"] == "MR"
    assert cap["area_desc"] == "Mumbai district of Maharashtra"
    assert cap["refs"] == ["IN-0_46", "IN-9"]
    assert cap["lgd_codes"] == ["482"] and cap["severity"] == "Severe"
    assert cap["sent"] == datetime(2026, 7, 20, 4, 30, tzinfo=UTC)
    assert cap["polygon_url"] == "https://example.invalid/poly?identifier=1"


def test_polygons_are_lon_lat_and_closed():
    rings = sachet.parse_polygon_file(POLYGON_FILE)
    assert rings[0][0] == (72.80, 18.90) and len(rings) == 2
    wkt = sachet.multipolygon_wkt(rings[:1])
    assert wkt.startswith("MULTIPOLYGON(((72.8 18.9, ") and wkt.endswith("72.8 18.9)))")
    assert sachet.multipolygon_wkt([]) is None


def test_lgd_codes_cover_all_24_wards_once():
    wards = sachet.CITY + sachet.SUBURBAN
    assert len(wards) == len(set(wards)) == 24


# --- With the database ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def db():
    from api.db import SessionLocal, init_db
    from api.db import engine as db_engine

    try:
        init_db()
        with db_engine.connect() as conn:
            n = conn.execute(text("SELECT count(*) FROM wards")).scalar()
    except OperationalError:
        pytest.skip("database not running")
    if n != 24:
        pytest.skip("wards not loaded")
    tag = f"TESTFIX-{secrets.token_hex(3)}"
    yield SessionLocal, db_engine, tag
    with db_engine.begin() as conn:
        conn.execute(text("DELETE FROM official_alerts WHERE identifier LIKE :t"), {"t": f"{tag}%"})
        conn.execute(text("DELETE FROM official_feed_fetches WHERE feed = :t"), {"t": tag})


def save(db, identifier: str, xml: bytes, rings=None, feed="maharashtra"):
    session_factory, _, _ = db
    cap = sachet.parse_cap(xml)
    with session_factory.begin() as session:
        item = {"guid": identifier, "agency": "IMD Mumbai", "link": None}
        sachet.store_alert(session, cap, item, feed, rings or [], datetime.now(UTC))


def wards_of(db, identifier: str) -> list[str]:
    with db[1].connect() as conn:
        return (
            conn.execute(
                text("SELECT ward_id FROM official_alert_wards WHERE identifier = :i ORDER BY 1"),
                {"i": identifier},
            )
            .scalars()
            .all()
        )


def test_wards_matched_and_updates_replace_older_messages(db):
    from api import main

    tag = db[2]
    first, update, cancel, lgd = (f"{tag}-{n}" for n in ("a", "b", "c", "d"))
    # Far in the future, so no real alert is in force at these times.
    t = datetime(2032, 7, 1, 4, 30, tzinfo=UTC)
    iso = lambda d: d.isoformat()  # noqa: E731
    box = sachet.parse_polygon_file(POLYGON_FILE)[:1]  # south Mumbai
    save(db, first, cap_xml(first, iso(t), iso(t + timedelta(hours=6))), box)
    assert wards_of(db, first) == ["A", "B", "C", "D", "E", "F-S", "G-S"]

    # The update arrives first: storing the older message afterwards still links them.
    later = t + timedelta(hours=1)
    save(db, update, cap_xml(update, iso(later), iso(t + timedelta(hours=6)), "Update",
                             f"IMD-Mumbai,{first},{iso(t)}"), box)  # fmt: skip
    with db[1].connect() as conn:
        assert (
            conn.execute(
                text("SELECT superseded_by FROM official_alerts WHERE identifier = :i"),
                {"i": first},
            ).scalar()
            == update
        )

    end = t + timedelta(hours=2)
    save(db, cancel, cap_xml(cancel, iso(end), iso(t + timedelta(hours=6)), "Cancel",
                             f"IMD-Mumbai,{update},{iso(later)}"), box)  # fmt: skip

    # No polygon: the LGD code for Mumbai Suburban picks its 15 wards.
    save(db, lgd, cap_xml(lgd, iso(t), iso(t + timedelta(hours=1)), lgd=("483", "497")))
    assert wards_of(db, lgd) == sorted(sachet.SUBURBAN)

    client = TestClient(main.app)

    def in_force(when):
        body = client.get("/v1/official-alerts", params={"at": iso(when)}).json()
        return sorted(a["identifier"] for a in body["alerts"] if a["identifier"].startswith(tag))

    assert in_force(t + timedelta(minutes=10)) == [first, lgd]
    assert in_force(later + timedelta(minutes=10)) == [update]  # lgd expired, first replaced
    assert in_force(end + timedelta(minutes=10)) == []  # cancelled
    body = client.get(
        "/v1/official-alerts", params={"at": iso(t + timedelta(minutes=10)), "ward_id": "T"}
    ).json()
    mine = [a for a in body["alerts"] if a["identifier"].startswith(tag)]
    assert [a["identifier"] for a in mine] == [lgd]
    assert mine[0]["colour"] == "orange" and mine[0]["match"] == "lgd"
    assert mine[0]["headline_local"] == "मुंबईत मुसळधार पाऊस"


def test_poll_stores_new_items_once(db, monkeypatch):
    tag = db[2]
    guid = f"{tag}-poll"
    feed = FEED.replace(b'<guid isPermaLink="false">111</guid>', f"<guid>{guid}</guid>".encode())
    feed = feed.replace(b"<guid>333</guid>", f"<guid>{tag}-incois</guid>".encode())

    class Resp:
        def __init__(self, content: bytes, url: str):
            self.content, self.url = content, url

    calls = []

    def fake_get(url, params=None, **_):
        calls.append(url)
        if url == "feed":
            return Resp(feed, url)
        if url == sachet.CAP_URL:
            now = datetime.now(UTC)
            ident = f"{tag}-cap-{params['identifier']}"
            return Resp(
                cap_xml(ident, now.isoformat(), (now + timedelta(hours=1)).isoformat()), url
            )
        return Resp(POLYGON_FILE, url)

    monkeypatch.setattr(sachet, "FEEDS", {tag: "feed"})
    monkeypatch.setattr(sachet.store, "http_get", fake_get)
    monkeypatch.setattr(sachet, "_save", lambda *a: None)  # keep fixtures out of the raw zone
    monkeypatch.setattr(sachet.store, "latest_raw", lambda *a: None)
    first = sachet.poll()
    # Read like the India feed: the item naming Mumbai and the INCOIS one, not Tripura's.
    assert first[tag]["error"] is None and first[tag]["new"] == 2
    assert wards_of(db, f"{tag}-cap-{guid}") == ["A", "B", "C", "D", "E", "F-S", "G-S"]
    calls.clear()
    again = sachet.poll()
    assert again[tag]["new"] == 0 and calls == ["feed"]  # nothing fetched twice
