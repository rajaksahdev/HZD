import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";

// Severity colours. Flood bands match SEVERITY_BANDS in models/risk.py; heat bands are the
// ERA5 1991-2020 percentiles in models/heat.py. Both provisional until Stage 13.
const LEVELS = [
  { key: "red", label: "Red", color: "#c62828" },
  { key: "orange", label: "Orange", color: "#ef6c00" },
  { key: "yellow", label: "Yellow", color: "#f9a825" },
  { key: "green", label: "Green", color: "#2e7d32" },
];
const HAZARDS = {
  flood: {
    title: "flood risk",
    ranges: { red: "probability 0.75 and above", orange: "0.50 to 0.75", yellow: "0.25 to 0.50", green: "below 0.25" },
  },
  heat: {
    title: "heat stress",
    ranges: {
      red: "99th percentile and above",
      orange: "95th to 99th",
      yellow: "90th to 95th",
      green: "below the 90th",
    },
  },
};
const NO_DATA = "#9e9e9e";
// Stage 12: facilities and routes. Hospitals in purple so they never read as a severity.
const PLACES = {
  shelter: { label: "Shelter candidate (school, community centre)", color: "#1565c0" },
  hospital: { label: "Hospital", color: "#6a1b9a" },
};
const UNSAFE = "#c62828"; // a route that crosses roads likely to flood
const EMPTY = { type: "FeatureCollection", features: [] };
const COLOR = Object.fromEntries(LEVELS.map((l) => [l.key, l.color]));
const STALE_MINUTES = 180; // same as api/main.py
const REFRESH_MS = 5 * 60 * 1000;
const TILES = import.meta.env.VITE_TILE_URL || "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

const ist = (iso, withTime = true) =>
  new Date(iso).toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    day: "2-digit",
    month: "short",
    year: "numeric",
    ...(withTime ? { hour: "2-digit", minute: "2-digit", hour12: false } : {}),
  });

const km = (m) => (m == null ? "—" : `${(m / 1000).toFixed(1)} km`);
const people = (n) =>
  n == null ? "—" : n >= 100000 ? `${(n / 100000).toFixed(1)} lakh` : n.toLocaleString("en-IN");

function routeText(r) {
  if (!r) return "not computed";
  const where = r.facility_name || "unnamed";
  if (r.status === "no_safe_route")
    return `no safe route: every way out crosses roads likely to flood (usual: ${where}, ${km(r.baseline_m)})`;
  const longer = r.baseline_m != null && r.distance_m > r.baseline_m ? `, usually ${km(r.baseline_m)}` : "";
  return `${where}, ${km(r.distance_m)} by road${longer}`;
}

function ageText(minutes) {
  if (minutes < 120) return `${minutes} min`;
  if (minutes < 48 * 60) return `${Math.floor(minutes / 60)} h`;
  return `${Math.floor(minutes / 1440)} days`;
}

// Where a run's rain came from (risk_runs.rain_source), in words.
function rainLabel(source, short = false) {
  const lead = /^forecast-lead(\d+)d$/.exec(source || "");
  if (lead) {
    const n = Number(lead[1]);
    const when = n === 0 ? "at 00:00 IST on the day" : `${n} day${n > 1 ? "s" : ""} ahead`;
    return short ? `replay, ${n === 0 ? "on the day" : when}` : `Archived forecast issued ${when} (replay)`;
  }
  if (source === "observed (replay)")
    return short ? "replay, observed" : "IMD observed, used as a perfect forecast (replay)";
  if (source === "observed") return short ? "observed" : "IMD observed (hindcast)";
  if (source === "forecast") return short ? "forecast" : "Corrected forecast";
  if (source === "era5") return short ? "ERA5" : "ERA5 reanalysis (hindcast)";
  return source;
}

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.detail || `${resp.status} from ${url}`);
  return body;
}

export default function App() {
  const mapBox = useRef(null);
  const map = useRef(null);
  const [runs, setRuns] = useState([]);
  // ?hazard=heat&run=<run_id> opens a given run, so a past event can be linked to directly.
  const [hazard, setHazard] = useState(() =>
    new URLSearchParams(location.search).get("hazard") === "heat" ? "heat" : "flood",
  );
  const [runId, setRunId] = useState(() => new URLSearchParams(location.search).get("run") || "");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [picked, setPicked] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [ready, setReady] = useState(null); // the map, once its layers exist
  const [impact, setImpact] = useState(null); // GET /v1/impact for the shown run
  const [wardImpact, setWardImpact] = useState(null); // the picked ward's impact and routes

  const load = useCallback(async () => {
    try {
      const q = new URLSearchParams({ hazard });
      if (runId) q.set("run_id", runId);
      const [runList, wards, people] = await Promise.all([
        getJson(`/v1/runs?hazard=${hazard}&limit=50`),
        getJson(`/v1/wards?${q}`),
        getJson(`/v1/impact?${q}`).catch(() => null),
      ]);
      setRuns(runList);
      setData(wards);
      setImpact(people);
      setError("");
    } catch (e) {
      setError(`Cannot load risk: ${e.message}`);
    }
  }, [runId, hazard]);

  useEffect(() => {
    const url = new URL(location.href);
    if (runId) url.searchParams.set("run", runId);
    else url.searchParams.delete("run");
    if (hazard === "heat") url.searchParams.set("hazard", "heat");
    else url.searchParams.delete("hazard");
    history.replaceState(null, "", url);
  }, [runId, hazard]);

  const switchHazard = (next) => {
    if (next === hazard) return;
    setHazard(next);
    setRunId("");
    setPicked(null);
    setWardImpact(null);
  };

  useEffect(() => {
    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => clearInterval(timer);
  }, [load]);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60 * 1000);
    return () => clearInterval(timer);
  }, []);

  // Create the map once.
  useEffect(() => {
    const m = new maplibregl.Map({
      container: mapBox.current,
      center: [72.87, 19.08],
      zoom: 10.2,
      style: {
        version: 8,
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
        sources: {
          base: {
            type: "raster",
            tiles: [TILES],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [{ id: "base", type: "raster", source: "base" }],
      },
    });
    m.addControl(new maplibregl.NavigationControl(), "top-right");
    m.on("load", () => {
      m.addSource("wards", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      m.addLayer({
        id: "ward-fill",
        type: "fill",
        source: "wards",
        paint: {
          "fill-color": ["match", ["get", "severity"], ...LEVELS.flatMap((l) => [l.key, l.color]), NO_DATA],
          "fill-opacity": 0.45,
        },
      });
      m.addLayer({
        id: "ward-line",
        type: "line",
        source: "wards",
        paint: { "line-color": "#263238", "line-width": 1 },
      });
      m.addLayer({
        id: "ward-label",
        type: "symbol",
        source: "wards",
        layout: { "text-field": ["get", "ward_id"], "text-font": ["Open Sans Semibold"], "text-size": 12 },
        paint: { "text-color": "#102027", "text-halo-color": "#ffffff", "text-halo-width": 1.5 },
      });
      m.addSource("routes", { type: "geojson", data: EMPTY });
      const unsafe = ["==", ["get", "status"], "no_safe_route"];
      m.addLayer({
        id: "route-line",
        type: "line",
        source: "routes",
        filter: ["!", unsafe],
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": ["match", ["get", "kind"], "hospital", PLACES.hospital.color, PLACES.shelter.color],
          "line-width": 4,
        },
      });
      m.addLayer({
        id: "route-unsafe",
        type: "line",
        source: "routes",
        filter: unsafe,
        paint: { "line-color": UNSAFE, "line-width": 4, "line-dasharray": [1, 1.5] },
      });
      m.addSource("facilities", { type: "geojson", data: EMPTY });
      m.addLayer({
        id: "facility-dot",
        type: "circle",
        source: "facilities",
        minzoom: 12,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 2.5, 16, 6],
          "circle-color": ["match", ["get", "group"], "hospital", PLACES.hospital.color, PLACES.shelter.color],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 1,
        },
      });
      m.on("mouseenter", "ward-fill", () => (m.getCanvas().style.cursor = "pointer"));
      m.on("mouseleave", "ward-fill", () => (m.getCanvas().style.cursor = ""));
      setReady(m);
    });
    map.current = m;
    return () => m.remove();
  }, []);

  // Clicking anywhere asks the API which ward the point is in (GET /v1/risk).
  useEffect(() => {
    if (!ready) return undefined;
    const onClick = async (e) => {
      const { lat, lng } = e.lngLat;
      const q = new URLSearchParams({ lat: lat.toFixed(5), lon: lng.toFixed(5), hazard });
      if (runId) q.set("run_id", runId);
      try {
        setPicked({ loading: true });
        setWardImpact(null);
        setPicked(await getJson(`/v1/risk?${q}`));
      } catch (err) {
        setPicked({ error: err.message });
      }
    };
    ready.on("click", onClick);
    return () => ready.off("click", onClick);
  }, [ready, runId, hazard]);

  useEffect(() => {
    if (ready && data) ready.getSource("wards").setData(data);
  }, [ready, data]);

  // Facilities never change during a session: load them once.
  useEffect(() => {
    if (!ready) return;
    getJson("/v1/facilities")
      .then((f) => ready.getSource("facilities").setData(f))
      .catch(() => {});
  }, [ready]);

  // The picked ward's people, shelter and hospital, for the run shown.
  const pickedWard = picked?.ward_id;
  useEffect(() => {
    if (!pickedWard) return;
    const q = new URLSearchParams({ hazard });
    if (runId) q.set("run_id", runId);
    getJson(`/v1/wards/${pickedWard}/impact?${q}`)
      .then(setWardImpact)
      .catch(() => setWardImpact(null));
  }, [pickedWard, runId, hazard]);

  useEffect(() => {
    if (ready) ready.getSource("routes").setData(wardImpact?.routes || EMPTY);
  }, [ready, wardImpact]);

  const run = data?.run;
  const ageMinutes = run ? Math.max(0, Math.floor((now - new Date(run.data_time)) / 60000)) : null;
  const wards = (data?.features || []).map((f) => f.properties);
  const isHeat = hazard === "heat";
  const score = (w) => (isHeat ? w.value : w.probability) ?? -Infinity;
  const ranked = [...wards].sort((a, b) => score(b) - score(a));
  const isAlarm = (w) => /^\w+ risk (RED|ORANGE):/.test(w);
  const alarming = (data?.warnings || []).filter(isAlarm);
  const otherWarnings = [
    ...(data?.warnings || []).filter((w) => !isAlarm(w)),
    ...(impact?.warnings || []),
  ];
  const impactOf = Object.fromEntries((impact?.wards || []).map((w) => [w.ward_id, w]));
  const mine = wardImpact?.wards?.[0];
  const ranges = HAZARDS[hazard].ranges;

  return (
    <div className="layout">
      <div className="map-wrap">
        <div ref={mapBox} className="map" />
        {alarming.length > 0 && (
          <div className="alarm" role="alert">
            {run?.is_test && <span className="badge test">TEST</span>}
            {alarming.map((w) => (
              <div key={w}>{w}</div>
            ))}
          </div>
        )}
      </div>

      <aside className="panel">
        <header>
          <h1>HazardLens {HAZARDS[hazard].title}</h1>
          {run?.is_test && <span className="badge test">TEST</span>}
        </header>

        <div className="toggle" role="group" aria-label="Hazard">
          {Object.keys(HAZARDS).map((h) => (
            <button key={h} type="button" aria-pressed={h === hazard} onClick={() => switchHazard(h)}>
              {h === "flood" ? "Flood" : "Heat"}
            </button>
          ))}
        </div>

        {error && <p className="error">{error}</p>}

        <label className="field">
          Run
          <select value={runId} onChange={(e) => setRunId(e.target.value)}>
            <option value="">Current (newest run valid now)</option>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.valid_from ? ist(r.valid_from, false) : "?"} · {rainLabel(r.rain_source, true)}
                {r.is_test ? " · TEST" : ""}
              </option>
            ))}
          </select>
        </label>

        {run && (
          <dl className="facts">
            <dt>Valid for</dt>
            <dd>{ist(run.valid_from, false)} (IST day)</dd>
            <dt>{isHeat ? "Inputs" : "Rain from"}</dt>
            <dd>{rainLabel(run.rain_source)}</dd>
            {isHeat && run.note && (
              <>
                <dt>Levels</dt>
                <dd>{run.note}</dd>
              </>
            )}
            <dt>Data age</dt>
            <dd className={ageMinutes > STALE_MINUTES ? "stale" : ""}>
              {ageText(ageMinutes)} (newest input {ist(run.data_time)})
            </dd>
            <dt>Issued</dt>
            <dd>{ist(run.issued_at)}</dd>
          </dl>
        )}

        {otherWarnings.length > 0 && (
          <ul className="warnings">
            {otherWarnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        )}

        <section>
          <h2>Picked point</h2>
          {!picked && <p className="muted">Click the map to see the risk for that place.</p>}
          {picked?.loading && <p className="muted">Looking up…</p>}
          {picked?.error && <p className="muted">{picked.error}</p>}
          {picked?.ward_id && (
            <div className="picked">
              <strong>Ward {picked.name}</strong>
              {picked.risks.length === 0 && <span className="muted"> no risk in this run</span>}
              {picked.risks.map((r) => (
                <span key={r.hazard} className="chip" style={{ background: COLOR[r.severity] }}>
                  {r.hazard} {r.severity} ·{" "}
                  {r.value != null ? `${r.value.toFixed(1)} ${r.unit || ""}` : `p ${r.probability.toFixed(2)}`} ·
                  trust {r.trust}
                </span>
              ))}
              {mine && (
                <dl className="facts impact">
                  <dt>People</dt>
                  <dd>
                    {people(mine.population)}
                    {mine.exposed != null && !isHeat && (
                      <span className="muted"> · {people(mine.exposed)} on low-lying land</span>
                    )}
                  </dd>
                  <dt>At risk</dt>
                  <dd>
                    {people(mine.people_at_risk)}
                    {mine.impact_score != null && (
                      <span className="muted"> · impact score {people(mine.impact_score)}</span>
                    )}
                  </dd>
                  <dt>Vulnerability</dt>
                  <dd>
                    {mine.vulnerability_index != null ? mine.vulnerability_index.toFixed(2) : "—"}{" "}
                    <span className="muted">(0 to 1, rank among the 24 wards)</span>
                  </dd>
                  <dt>
                    <span className="dot" style={{ background: PLACES.shelter.color }} />
                    Shelter
                  </dt>
                  <dd>{routeText(mine.shelter)}</dd>
                  <dt>
                    <span className="dot" style={{ background: PLACES.hospital.color }} />
                    Hospital
                  </dt>
                  <dd>{routeText(mine.hospital)}</dd>
                  {mine.shelter?.avoided_km > 0 && (
                    <>
                      <dt>Avoided</dt>
                      <dd>
                        {mine.shelter.avoided_km.toFixed(1)} km of low-lying or waterlogging-prone road
                        {mine.shelter.people_cut_off > 0 &&
                          `; ${people(mine.shelter.people_cut_off)} people cannot reach a shelter without using one`}
                      </dd>
                    </>
                  )}
                </dl>
              )}
            </div>
          )}
        </section>

        <section>
          <h2>Severity</h2>
          <ul className="legend">
            {LEVELS.map((l) => (
              <li key={l.key}>
                <span className="swatch" style={{ background: l.color }} />
                {l.label} <span className="muted">{ranges[l.key]}</span>
              </li>
            ))}
            <li>
              <span className="swatch" style={{ background: NO_DATA }} />
              No data
            </li>
          </ul>
          <ul className="legend">
            {Object.entries(PLACES).map(([key, p]) => (
              <li key={key}>
                <span className="dot" style={{ background: p.color }} />
                {p.label} <span className="muted">(zoom in)</span>
              </li>
            ))}
            <li>
              <span className="dash" style={{ borderColor: UNSAFE }} />
              Route that crosses roads likely to flood
            </li>
          </ul>
        </section>

        <section>
          <h2>Wards</h2>
          <table className="wards">
            <thead>
              <tr>
                <th>Ward</th>
                <th>Level</th>
                <th className="num">{isHeat ? "Feels like (max)" : "Probability"}</th>
                <th className="num" title="Probability x people exposed">
                  At risk
                </th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((w) => (
                <tr key={w.ward_id}>
                  <td>{w.name}</td>
                  <td>
                    <span className="swatch" style={{ background: COLOR[w.severity] || NO_DATA }} />
                    {w.severity || "—"}
                  </td>
                  <td className="num">
                    {isHeat
                      ? w.value != null
                        ? `${w.value.toFixed(1)} ${w.unit || ""}`
                        : "—"
                      : w.probability != null
                        ? w.probability.toFixed(2)
                        : "—"}
                  </td>
                  <td className="num">{people(impactOf[w.ward_id]?.people_at_risk)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <footer className="muted">{data?.disclaimer}</footer>
      </aside>
    </div>
  );
}
