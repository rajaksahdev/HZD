# Alert engine (Stage 13)

Built 21 Nov 2026 with `python -m alerts.engine --targets --report docs/alerts.md`. Method in `alerts/engine.py`; signing in `alerts/keys.py`; CAP in `alerts/cap.py`; operators in `alerts/operators.py`.

## Severity rules

| Level in the run | Flood action code | Heat action code | Sent |
| --- | --- | --- | --- |
| green, yellow | – | – | no alert; an active alert is cancelled |
| orange | FLOOD_AVOID_LOW_ROADS | HEAT_LIMIT_SUN | at once |
| red | FLOOD_MOVE_UP | HEAT_STAY_COOL | after an operator approves |

Also held for an operator: an update that lowers a red alert, a cancel that ends one, a new alert with low trust (the forecast models disagree, Stage 11) and a new alert in a ward over its false-alarm target. The levels come from the risk runs: flood probability bands 0.25 / 0.5 / 0.75 (Stage 6), heat percentiles 90 / 95 / 99 (Stage 9). Landslide susceptibility (Stage 10) is a static map and sends no alerts.

## Messages

- One alert per ward and hazard episode (`alert_id`, e.g. `flood-F-N-20250819-3fa2c1`). Its messages count up from `sequence` 1: alert, updates, at most one cancel. A receiver drops a message whose sequence is not above the last it holds, so an old message replayed later changes nothing.
- Every message expires: at the end of the risk window it is about (a cancel: six hours after the alert would have expired).
- Ed25519 signature over the JSON without `signature`, keys sorted, no spaces, UTF-8. `key_id` names the public key (`GET /v1/alert-keys`). The private key is a file on the machine that runs the engine (`ALERT_SIGNING_KEY_PATH`); the API never opens it. In the pilot both run on one laptop: the split is in the code, not yet in hardware.
- Every message is TEST unless the run is live and `ALERTS_LIVE=1` is set.
- CAP 1.2: `GET /v1/alerts/{alert_id}/cap`. Status Test or Actual, msgType Alert, Update or Cancel with `<references>` to the earlier messages, the ward outline as polygons, BMC ward geocode, and the signature, key_id and sequence as parameters.

## Operators

Approval needs a password (scrypt hash) and a six-digit authenticator code (TOTP, RFC 6238). A code works once; five failures lock the operator for 15 minutes. Operators can also cancel any active alert by hand. The command line is the only approval screen until the Stage 22 dashboard.

## False-alarm target per ward

Target: at most 50% of a ward's flood alerts unconfirmed by a flood report, judged once a ward has 5 or more alert-days. A ward over its target has its new orange alerts held for an operator. Set a different target with `--set-target WARD flood SHARE`.

Measured on June to September 2024 and 2025, day-ahead runs: every day-ahead run from the Stage 7 forecast archive, one alert-day for each ward at orange or red. The 14 reported flooded ward-days in those seasons are the only ground truth, so an unconfirmed day is not proof of a false alarm: most days have no report either way. The heavy-rain rule (Stage 5 baseline) is shown for comparison.

All wards: the model alerted on 3404 ward-days, 14 confirmed (99.6% unconfirmed); the rule on 632, 10 confirmed (98.4% unconfirmed). 24 of 24 wards are over the target.

| Ward | Days | Reported floods | Model alert-days | Confirmed | Unconfirmed | Rule days | Rule unconfirmed | Over target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 244 | 0 | 154 | 0 | 100% | 29 | 100% | yes |
| B | 244 | 0 | 153 | 0 | 100% | 28 | 100% | yes |
| C | 244 | 0 | 156 | 0 | 100% | 28 | 100% | yes |
| D | 244 | 0 | 154 | 0 | 100% | 28 | 100% | yes |
| E | 244 | 1 | 152 | 1 | 99% | 29 | 97% | yes |
| F-N | 244 | 2 | 142 | 2 | 99% | 28 | 96% | yes |
| F-S | 244 | 2 | 146 | 2 | 99% | 28 | 96% | yes |
| G-N | 244 | 1 | 142 | 1 | 99% | 28 | 96% | yes |
| G-S | 244 | 1 | 147 | 1 | 99% | 29 | 97% | yes |
| H-E | 244 | 0 | 139 | 0 | 100% | 27 | 100% | yes |
| H-W | 244 | 1 | 139 | 1 | 99% | 27 | 96% | yes |
| K-E | 244 | 2 | 139 | 2 | 99% | 24 | 96% | yes |
| K-W | 244 | 1 | 139 | 1 | 99% | 24 | 96% | yes |
| L | 244 | 1 | 139 | 1 | 99% | 26 | 96% | yes |
| M-E | 244 | 0 | 141 | 0 | 100% | 29 | 100% | yes |
| M-W | 244 | 1 | 142 | 1 | 99% | 28 | 96% | yes |
| N | 244 | 0 | 137 | 0 | 100% | 25 | 100% | yes |
| P-N | 244 | 1 | 137 | 1 | 99% | 24 | 100% | yes |
| P-S | 244 | 0 | 138 | 0 | 100% | 24 | 100% | yes |
| R-C | 244 | 0 | 135 | 0 | 100% | 24 | 100% | yes |
| R-N | 244 | 0 | 134 | 0 | 100% | 23 | 100% | yes |
| R-S | 244 | 0 | 136 | 0 | 100% | 24 | 100% | yes |
| S | 244 | 0 | 134 | 0 | 100% | 24 | 100% | yes |
| T | 244 | 0 | 129 | 0 | 100% | 24 | 100% | yes |

What this means: 24 of 24 wards are over the target. Two things add up. The Stage 5 flood model reaches orange on most monsoon days (its labels have only dry-day negatives), 5 times as often as the heavy-rain rule. And reports are so few that even the rule leaves almost every alert-day unconfirmed. Counting days without a report as false alarms is the cautious reading: over the target, new flood alerts wait for an operator rather than go out unchecked. Once citizen reports (Stage 19) give outcomes, including 'no flooding here', the share should be measured over alert-days with an outcome; the model fix is hard negatives from BMC records.

## Example message

A TEST flood alert as receivers get it (signed with a throwaway key for this page):

```xml
<?xml version='1.0' encoding='utf-8'?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>hazardlens-flood-F-N-20250819-3fa2c1-1</identifier>
  <sender>hazardlens.prototype</sender>
  <sent>2025-08-18T14:00:00+05:30</sent>
  <status>Test</status>
  <msgType>Alert</msgType>
  <source>HazardLens run replay-20250819-lead1d</source>
  <scope>Public</scope>
  <note>TEST message from a prototype. Take no action.</note>
  <incidents>flood-F-N-20250819-3fa2c1</incidents>
  <info>
    <language>en-IN</language>
    <category>Met</category>
    <event>Urban flooding</event>
    <responseType>Avoid</responseType>
    <responseType>Prepare</responseType>
    <urgency>Expected</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <eventCode>
      <valueName>HazardLens</valueName>
      <value>FLOOD_AVOID_LOW_ROADS</value>
    </eventCode>
    <effective>2025-08-18T14:00:00+05:30</effective>
    <onset>2025-08-19T00:00:00+05:30</onset>
    <expires>2025-08-20T00:00:00+05:30</expires>
    <senderName>HazardLens (decision-support prototype, not an official warning)</senderName>
    <headline>TEST. Orange urban flooding alert for ward F/N</headline>
    <instruction>Avoid low-lying roads, subways and underpasses. Keep documents, medicines and a charged phone ready.</instruction>
    <parameter>
      <valueName>HazardLens:alert_id</valueName>
      <value>flood-F-N-20250819-3fa2c1</value>
    </parameter>
    <parameter>
      <valueName>HazardLens:sequence</valueName>
      <value>1</value>
    </parameter>
    <parameter>
      <valueName>HazardLens:trust</valueName>
      <value>medium</value>
    </parameter>
    <parameter>
      <valueName>HazardLens:key_id</valueName>
      <value>hzd-e7e37c40b0cbfef7</value>
    </parameter>
    <parameter>
      <valueName>HazardLens:signature_ed25519</valueName>
      <value>TmlzMT9Sxu0/q9IAU8bgHrNpZfG8+5tCDPdWD7FigxQ6b0jO1jtbXbsbOOyYnI8gQaUkE3ywKyenqsl7lRnqCA==</value>
    </parameter>
    <area>
      <areaDesc>Mumbai, BMC ward F/N</areaDesc>
      <geocode>
        <valueName>BMC ward</valueName>
        <value>F-N</value>
      </geocode>
    </area>
  </info>
</alert>
```

## Limits

- The false-alarm share counts days without a report as unconfirmed. Citizen reports (Stage 19) and the feedback loop (Stage 20) will add outcomes; until then the share is an upper bound.
- Heat has no outcome data, so heat targets are set but not measured.
- The English headline and instruction in CAP are placeholders for the reviewed templates in three languages (Stage 16). Delivery (Telegram, web push, SMS) is Stage 18.
- The private key file is unencrypted on disk; protect it by file permissions and keep it off any machine that serves the public API.
