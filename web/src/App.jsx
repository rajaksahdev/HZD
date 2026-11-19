import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";

// Severity colours. The bands match SEVERITY_BANDS in models/risk.py (provisional until Stage 13).
const LEVELS = [
  { key: "red", label: "Red", range: "0.75 and above", color: "#c62828" },
  { key: "orange", label: "Orange", range: "0.50 to 0.75", color: "#ef6c00" },
  { key: "yellow", label: "Yellow", range: "0.25 to 0.50", color: "#f9a825" },
  { key: "green", label: "Green", range: "below 0.25", color: "#2e7d32" },
];
const NO_DATA = "#9e9e9e";
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

function ageText(minutes) {
  if (minutes < 120) return `${minutes} min`;
  if (minutes < 48 * 60) return `${Math.floor(minutes / 60)} h`;
  return `${Math.floor(minutes / 1440)} days`;
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
  // ?run=<run_id> opens a given run, so a past event can be linked to directly.
  const [runId, setRunId] = useState(() => new URLSearchParams(location.search).get("run") || "");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [picked, setPicked] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [ready, setReady] = useState(null); // the map, once its layers exist

  const load = useCallback(async () => {
    try {
      const [runList, wards] = await Promise.all([
        getJson("/v1/runs"),
        getJson(runId ? `/v1/wards?run_id=${encodeURIComponent(runId)}` : "/v1/wards"),
      ]);
      setRuns(runList);
      setData(wards);
      setError("");
    } catch (e) {
      setError(`Cannot load risk: ${e.message}`);
    }
  }, [runId]);

  useEffect(() => {
    const url = new URL(location.href);
    if (runId) url.searchParams.set("run", runId);
    else url.searchParams.delete("run");
    history.replaceState(null, "", url);
  }, [runId]);

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
      const q = new URLSearchParams({ lat: lat.toFixed(5), lon: lng.toFixed(5) });
      if (runId) q.set("run_id", runId);
      try {
        setPicked({ loading: true });
        setPicked(await getJson(`/v1/risk?${q}`));
      } catch (err) {
        setPicked({ error: err.message });
      }
    };
    ready.on("click", onClick);
    return () => ready.off("click", onClick);
  }, [ready, runId]);

  useEffect(() => {
    if (ready && data) ready.getSource("wards").setData(data);
  }, [ready, data]);

  const run = data?.run;
  const ageMinutes = run ? Math.max(0, Math.floor((now - new Date(run.data_time)) / 60000)) : null;
  const wards = (data?.features || []).map((f) => f.properties);
  const ranked = [...wards].sort((a, b) => (b.probability ?? -1) - (a.probability ?? -1));
  const alarming = (data?.warnings || []).filter((w) => w.startsWith("Flood risk"));
  const otherWarnings = (data?.warnings || []).filter((w) => !w.startsWith("Flood risk"));

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
          <h1>HazardLens flood risk</h1>
          {run?.is_test && <span className="badge test">TEST</span>}
        </header>

        {error && <p className="error">{error}</p>}

        <label className="field">
          Run
          <select value={runId} onChange={(e) => setRunId(e.target.value)}>
            <option value="">Current (newest run valid now)</option>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.valid_from ? ist(r.valid_from, false) : "?"} · {r.rain_source}
                {r.is_test ? " · TEST" : ""}
              </option>
            ))}
          </select>
        </label>

        {run && (
          <dl className="facts">
            <dt>Valid for</dt>
            <dd>{ist(run.valid_from, false)} (IST day)</dd>
            <dt>Rain from</dt>
            <dd>{run.rain_source === "observed" ? "IMD observed (hindcast)" : "Corrected forecast"}</dd>
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
                  {r.hazard} {r.severity} · p {r.probability.toFixed(2)} · trust {r.trust}
                </span>
              ))}
            </div>
          )}
        </section>

        <section>
          <h2>Severity</h2>
          <ul className="legend">
            {LEVELS.map((l) => (
              <li key={l.key}>
                <span className="swatch" style={{ background: l.color }} />
                {l.label} <span className="muted">probability {l.range}</span>
              </li>
            ))}
            <li>
              <span className="swatch" style={{ background: NO_DATA }} />
              No data
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
                <th className="num">Probability</th>
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
                  <td className="num">{w.probability != null ? w.probability.toFixed(2) : "—"}</td>
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
