"""CAP 1.2 export of one alert message (Stage 13).

Common Alerting Protocol 1.2 (OASIS), the format SACHET and most national systems exchange.
Elements follow the order the CAP schema requires. Times are in IST with an explicit offset
(CAP forbids "Z"). The Ed25519 signature of the JSON message travels as parameters; the XML
itself carries no XML signature.

Each message is its own CAP message: identifier hazardlens-{alert_id}-{sequence}. An update or
cancel lists every earlier message of the alert in <references>.

The English headline and instruction are provisional: Stage 16 renders the action code from
reviewed templates in English, Hindi and Marathi.
"""

from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

from api.schemas import Alert, Hazard, MessageType, Severity, Trust

NS = "urn:oasis:names:tc:emergency:cap:1.2"
SENDER = "hazardlens.prototype"
SENDER_NAME = "HazardLens (decision-support prototype, not an official warning)"
IST = timezone(timedelta(hours=5, minutes=30))

EVENT = {Hazard.FLOOD: "Urban flooding", Hazard.HEAT: "Heat stress"}
CATEGORY = {Hazard.FLOOD: "Met", Hazard.HEAT: "Health"}
CAP_SEVERITY = {
    Severity.RED: "Severe",
    Severity.ORANGE: "Moderate",
    Severity.YELLOW: "Minor",
    Severity.GREEN: "Minor",
}
CERTAINTY = {Trust.HIGH: "Likely", Trust.MEDIUM: "Likely", Trust.LOW: "Possible"}
MSG_TYPE = {MessageType.ALERT: "Alert", MessageType.UPDATE: "Update", MessageType.CANCEL: "Cancel"}

# action_code -> (CAP responseType values, provisional English instruction)
ACTIONS = {
    "FLOOD_AVOID_LOW_ROADS": (
        ["Avoid", "Prepare"],
        "Avoid low-lying roads, subways and underpasses. Keep documents, medicines and a "
        "charged phone ready.",
    ),
    "FLOOD_MOVE_UP": (
        ["Prepare", "Evacuate"],
        "If water starts to enter, move people, documents and medicines to a higher floor or "
        "the nearest shelter. Do not walk or drive through flood water.",
    ),
    "HEAT_LIMIT_SUN": (
        ["Prepare"],
        "Avoid outdoor work from 12:00 to 16:00. Drink water often. Check on elderly neighbours.",
    ),
    "HEAT_STAY_COOL": (
        ["Shelter", "Prepare"],
        "Stay indoors or in shade from 11:00 to 17:00 and drink water often. Seek medical help "
        "for dizziness, confusion or hot dry skin.",
    ),
    "ALL_CLEAR": (["AllClear"], "The earlier alert for this ward has ended."),
}


def cap_time(when: datetime) -> str:
    return when.astimezone(IST).replace(microsecond=0).isoformat()


def identifier(alert_id: str, sequence: int) -> str:
    return f"hazardlens-{alert_id}-{sequence}"


def _sub(parent: ET.Element, tag: str, value: str | None = None) -> ET.Element:
    el = ET.SubElement(parent, f"{{{NS}}}{tag}")
    if value is not None:
        el.text = value
    return el


def polygon_text(ring: list[tuple[float, float]]) -> str:
    """CAP polygon: 'lat,lon' pairs separated by spaces, closed (first = last), 4+ points."""
    points = list(ring)
    if points[0] != points[-1]:
        points.append(points[0])
    if len(points) < 4:
        raise ValueError("a CAP polygon needs at least 4 points")
    return " ".join(f"{lat:.5f},{lon:.5f}" for lon, lat in points)


def headline(alert: Alert, ward_name: str) -> str:
    test = "TEST. " if alert.is_test else ""
    if alert.message_type == MessageType.CANCEL:
        return f"{test}{EVENT[alert.hazard]} alert for ward {ward_name} cancelled"
    level = alert.severity.value.capitalize()
    verb = "updated" if alert.message_type == MessageType.UPDATE else ""
    return f"{test}{level} {EVENT[alert.hazard].lower()} alert for ward {ward_name} {verb}".strip()


def to_cap(
    alert: Alert,
    ward_name: str,
    polygons: list[list[tuple[float, float]]] | None = None,
    earlier: list[Alert] | None = None,
) -> str:
    """CAP 1.2 XML for one signed message. `polygons`: the ward's outer rings as (lon, lat);
    `earlier`: the alert's previous messages, for <references>."""
    if alert.signature is None:
        raise ValueError("only signed messages are exported")
    ET.register_namespace("", NS)
    root = ET.Element(f"{{{NS}}}alert")
    _sub(root, "identifier", identifier(alert.alert_id, alert.sequence))
    _sub(root, "sender", SENDER)
    _sub(root, "sent", cap_time(alert.issued_at))
    _sub(root, "status", "Test" if alert.is_test else "Actual")
    _sub(root, "msgType", MSG_TYPE[alert.message_type])
    _sub(root, "source", f"HazardLens run {alert.run_id}" if alert.run_id else "HazardLens")
    _sub(root, "scope", "Public")
    if alert.is_test:
        _sub(root, "note", "TEST message from a prototype. Take no action.")
    if alert.message_type != MessageType.ALERT and earlier:
        refs = [
            f"{SENDER},{identifier(e.alert_id, e.sequence)},{cap_time(e.issued_at)}"
            for e in sorted(earlier, key=lambda e: e.sequence)
            if e.sequence < alert.sequence
        ]
        if refs:
            _sub(root, "references", " ".join(refs))
    _sub(root, "incidents", alert.alert_id)

    info = _sub(root, "info")
    _sub(info, "language", "en-IN")
    _sub(info, "category", CATEGORY[alert.hazard])
    _sub(info, "event", EVENT[alert.hazard])
    responses, instruction = ACTIONS.get(alert.action_code, (["Monitor"], None))
    for response in responses:
        _sub(info, "responseType", response)
    cancelled = alert.message_type == MessageType.CANCEL
    soon = alert.onset is None or alert.onset <= alert.issued_at
    _sub(info, "urgency", "Past" if cancelled else ("Immediate" if soon else "Expected"))
    _sub(info, "severity", CAP_SEVERITY[alert.severity])
    _sub(info, "certainty", "Unlikely" if cancelled else CERTAINTY[alert.trust])
    _sub(info, "eventCode").extend(_value_pair("HazardLens", alert.action_code))
    _sub(info, "effective", cap_time(alert.issued_at))
    if alert.onset is not None:
        _sub(info, "onset", cap_time(alert.onset))
    _sub(info, "expires", cap_time(alert.expires_at))
    _sub(info, "senderName", SENDER_NAME)
    _sub(info, "headline", headline(alert, ward_name))
    if instruction:
        _sub(info, "instruction", instruction)
    for name, value in (
        ("alert_id", alert.alert_id),
        ("sequence", str(alert.sequence)),
        ("trust", alert.trust.value),
        ("key_id", alert.key_id),
        ("signature_ed25519", alert.signature),
    ):
        _sub(info, "parameter").extend(_value_pair(f"HazardLens:{name}", value))

    area = _sub(info, "area")
    _sub(area, "areaDesc", f"Mumbai, BMC ward {ward_name}")
    for ring in polygons or []:
        _sub(area, "polygon", polygon_text(ring))
    _sub(area, "geocode").extend(_value_pair("BMC ward", alert.ward_id))

    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _value_pair(name: str, value: str) -> list[ET.Element]:
    a, b = ET.Element(f"{{{NS}}}valueName"), ET.Element(f"{{{NS}}}value")
    a.text, b.text = name, value
    return [a, b]
