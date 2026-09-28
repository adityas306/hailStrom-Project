import React, { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Circle,
  Popup,
  Polyline,
  useMap,
  useMapEvents,
} from "react-leaflet";

const API =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

function ClickHandler({ onClick }) {
  useMapEvents({
    click(e) {
      onClick(e.latlng);
    },
  });

  return null;
}

function MapController({ selected }) {
  const map = useMap();

  useEffect(() => {
    if (!selected) return;

    map.flyTo(
      [selected.lat, selected.lon],
      9,
      {
        duration: 1.2,
      }
    );
  }, [selected, map]);

  return null;
}

function riskColor(p = 0) {
  if (p >= 0.75) return "#ff3b30";
  if (p >= 0.45) return "#ffb020";
  return "#22c55e";
}

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const [selected, setSelected] = useState({
    name: "Lucknow, Uttar Pradesh",
    lat: 26.8467,
    lon: 80.9462,
  });

  const [hazard, setHazard] = useState(null);
  const [cells, setCells] = useState([]);
  const [track, setTrack] = useState([]);
  const [sourceMode, setSourceMode] = useState("CONNECTING");
  const [updated, setUpdated] = useState(null);
  const [locationLoading, setLocationLoading] = useState(false);

  // --------------------------------------------------
  // LOAD HAZARD FORECAST
  // --------------------------------------------------

  async function loadHazard(lat, lon, name) {
    try {
      setLocationLoading(true);

      const response = await fetch(
        `${API}/api/hazards?lat=${lat}&lon=${lon}`
      );

      if (!response.ok) {
        throw new Error("Hazard API failed");
      }

      const data = await response.json();

      setHazard({
        ...data,
        name,
      });

      setSelected({
        name,
        lat: Number(lat),
        lon: Number(lon),
      });
    } catch (error) {
      console.error("Hazard error:", error);

      setHazard(null);

      setSelected({
        name,
        lat: Number(lat),
        lon: Number(lon),
      });
    } finally {
      setLocationLoading(false);
    }
  }

  // --------------------------------------------------
  // REVERSE GEOCODING
  // MAP COORDINATES -> LOCATION NAME
  // --------------------------------------------------

  async function getLocationName(lat, lon) {
    try {
      const response = await fetch(
        `${API}/api/location/reverse?lat=${lat}&lon=${lon}`
      );

      if (!response.ok) {
        throw new Error("Reverse location API failed");
      }

      const data = await response.json();

      return (
        data.name ||
        data.display_name ||
        data.city ||
        data.location ||
        "Selected Location"
      );
    } catch (error) {
      console.error(
        "Reverse geocoding error:",
        error
      );

      return "Selected Location";
    }
  }

  // --------------------------------------------------
  // MAP CLICK
  // --------------------------------------------------

  async function mapClick(position) {
    try {
      setLocationLoading(true);

      const lat = position.lat;
      const lon = position.lng;

      // Get actual city/district/state name
      const name = await getLocationName(
        lat,
        lon
      );

      await loadHazard(
        lat,
        lon,
        name
      );
    } catch (error) {
      console.error(
        "Map click error:",
        error
      );

      setLocationLoading(false);
    }
  }

  // --------------------------------------------------
  // LOCATION SEARCH
  // --------------------------------------------------

  async function search() {
    if (!query.trim()) return;

    try {
      const response = await fetch(
        `${API}/api/location/search?q=${encodeURIComponent(
          query
        )}`
      );

      if (!response.ok) {
        throw new Error(
          "Location search failed"
        );
      }

      const data = await response.json();

      setResults(
        data.results || []
      );
    } catch (error) {
      console.error(
        "Search error:",
        error
      );

      setResults([]);
    }
  }

  // --------------------------------------------------
  // SELECT SEARCH RESULT
  // --------------------------------------------------

  function selectResult(location) {
    setResults([]);

    setQuery(
      location.name
    );

    loadHazard(
      Number(location.lat),
      Number(location.lon),
      location.name
    );
  }

  // --------------------------------------------------
  // INITIAL LOAD + WEBSOCKET
  // --------------------------------------------------

  useEffect(() => {
    loadHazard(
      selected.lat,
      selected.lon,
      selected.name
    );

    const wsUrl =
      API.replace(/^http/, "ws") +
      "/ws/weather";

    const ws =
      new WebSocket(wsUrl);

    ws.onopen = () => {
      setSourceMode(
        "LIVE STREAM"
      );
    };

    ws.onmessage = (event) => {
      try {
        const data =
          JSON.parse(
            event.data
          );

        setCells(
          data.cells || []
        );

        setTrack(
          (data.storm_track || [])
            .map((point) => [
              point.lat,
              point.lon,
            ])
        );

        setSourceMode(
          data.data_mode === "demo"
            ? "DEMO STREAM"
            : "LIVE"
        );

        if (data.timestamp) {
          setUpdated(
            new Date(
              data.timestamp
            )
          );
        }
      } catch (error) {
        console.error(
          "WebSocket error:",
          error
        );
      }
    };

    ws.onerror = () => {
      setSourceMode(
        "OFFLINE"
      );
    };

    return () => {
      ws.close();
    };

    // Initial load only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --------------------------------------------------
  // MAP CENTER
  // --------------------------------------------------

  const mapCenter = useMemo(
    () => [
      selected.lat,
      selected.lon,
    ],
    [
      selected.lat,
      selected.lon,
    ]
  );

  // --------------------------------------------------
  // FORECAST VALUES
  // --------------------------------------------------

  const arrival =
    hazard?.arrival_minutes ??
    "--";

  const countdown =
    arrival === "--"
      ? "--:--"
      : `00:${String(
          arrival
        ).padStart(2, "0")}:00`;

  return (
    <div className="app">

      {/* =========================================
          HEADER
      ========================================= */}

      <header className="topbar">
        <div>
          <div className="brand">
            ⛈ StormCast{" "}
            <span>India</span>
          </div>

          <div className="subtitle">
            AI Convective Storm
            Nowcasting • 0–6 hours
            • 1–3 km
          </div>
        </div>

        <div className="liveBadge">
          <i />
          {sourceMode}
        </div>
      </header>

      {/* =========================================
          SEARCH
      ========================================= */}

      <section className="searchRow">

        <div className="searchBox">

          <input
            value={query}
            onChange={(e) =>
              setQuery(
                e.target.value
              )
            }
            onKeyDown={(e) => {
              if (
                e.key === "Enter"
              ) {
                search();
              }
            }}
            placeholder="Search city / district / location..."
          />

          <button
            onClick={search}
          >
            Search
          </button>

        </div>

        {results.length > 0 && (
          <div className="results">

            {results.map(
              (
                location,
                index
              ) => (

                <button
                  key={`${location.name}-${index}`}
                  onClick={() =>
                    selectResult(
                      location
                    )
                  }
                >

                  <span>
                    {
                      location.name
                    }
                  </span>

                  <small>
                    {Number(
                      location.lat
                    ).toFixed(2)}
                    {" , "}
                    {Number(
                      location.lon
                    ).toFixed(2)}
                  </small>

                </button>

              )
            )}

          </div>
        )}

      </section>

      {/* =========================================
          MAIN DASHBOARD
      ========================================= */}

      <main className="grid">

        {/* =======================================
            MAP
        ======================================= */}

        <section className="mapCard">

          <div className="cardTitle">

            <span>
              LIVE GIS HAZARD MAP
            </span>

            <small>
              Click anywhere to inspect
            </small>

          </div>

          <div className="mapWrap">

            <MapContainer
              center={mapCenter}
              zoom={7}
              scrollWheelZoom={true}
            >

              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />

              {/* Map click listener */}

              <ClickHandler
                onClick={mapClick}
              />

              {/* Automatically move map */}

              <MapController
                selected={selected}
              />

              {/* =================================
                  HAZARD CELLS
              ================================= */}

              {cells.map(
                (
                  cell,
                  index
                ) => {

                  const probability =
                    cell.storm_probability ??
                    0;

                  return (
                    <Circle
                      key={index}
                      center={[
                        cell.lat,
                        cell.lon,
                      ]}
                      radius={
                        12000 +
                        probability *
                          16000
                      }
                      pathOptions={{
                        color:
                          riskColor(
                            probability
                          ),

                        fillColor:
                          riskColor(
                            probability
                          ),

                        fillOpacity:
                          0.34,

                        weight: 2,
                      }}
                    >

                      <Popup>

                        <b>
                          Hazard Grid
                        </b>

                        <br />

                        Storm:{" "}
                        {(
                          probability *
                          100
                        ).toFixed(0)}
                        %

                        <br />

                        Hail:{" "}
                        {(
                          (cell.hail_probability ??
                            0) *
                          100
                        ).toFixed(0)}
                        %

                        <br />

                        Cloudburst:{" "}
                        {(
                          (cell.cloudburst_probability ??
                            0) *
                          100
                        ).toFixed(0)}
                        %

                        <br />

                        Lightning:{" "}
                        {(
                          cell.lightning_density ??
                          0
                        ).toFixed(1)}
                        {" / 100 km²"}

                        <br />

                        Arrival:{" "}
                        {
                          cell.arrival_minutes ??
                          "--"
                        }{" "}
                        min

                      </Popup>

                    </Circle>
                  );
                }
              )}

              {/* =================================
                  STORM TRACK
              ================================= */}

              {track.length > 1 && (
                <Polyline
                  positions={track}
                  pathOptions={{
                    color:
                      "#6d28d9",

                    weight: 5,

                    dashArray:
                      "8 8",
                  }}
                />
              )}

              {/* =================================
                  SELECTED LOCATION
              ================================= */}

              {selected && (
                <Circle
                  center={[
                    selected.lat,
                    selected.lon,
                  ]}
                  radius={5000}
                  pathOptions={{
                    color:
                      "#111827",

                    fillOpacity:
                      0.05,

                    weight: 3,
                  }}
                >

                  <Popup>

                    <b>
                      📍{" "}
                      {
                        selected.name
                      }
                    </b>

                    <br />

                    Latitude:{" "}
                    {selected.lat.toFixed(
                      4
                    )}

                    <br />

                    Longitude:{" "}
                    {selected.lon.toFixed(
                      4
                    )}

                  </Popup>

                </Circle>
              )}

            </MapContainer>

          </div>

        </section>

        {/* =======================================
            SIDEBAR
        ======================================= */}

        <aside className="side">

          {/* =====================================
              SELECTED LOCATION
          ===================================== */}

          <div className="panel selected">

            <div className="eyebrow">
              SELECTED LOCATION
            </div>

            <h2>
              {locationLoading
                ? "Loading location..."
                : hazard?.name ||
                  selected.name}
            </h2>

            <div className="coords">

              {selected.lat.toFixed(
                4
              )}

              {" , "}

              {selected.lon.toFixed(
                4
              )}

            </div>

            <div
              className={`risk ${
                hazard?.risk?.toLowerCase() ||
                ""
              }`}
            >
              {locationLoading
                ? "LOADING"
                : hazard?.risk ||
                  "LOADING"}
            </div>

            <div className="countdown">

              <span>
                Expected storm arrival
              </span>

              <strong>
                {countdown}
              </strong>

              <small>

                {arrival !== "--"
                  ? `${arrival} minutes estimated`
                  : ""}

              </small>

            </div>

          </div>

          {/* =====================================
              HAZARD FORECAST
          ===================================== */}

          <div className="panel">

            <div className="eyebrow">
              HAZARD FORECAST
            </div>

            <Metric
              label="Storm probability"
              value={
                hazard
                  ? `${(
                      (hazard.storm_probability ??
                        0) *
                      100
                    ).toFixed(0)}%`
                  : "--"
              }
            />

            <Metric
              label="Lightning density"
              value={
                hazard &&
                hazard.lightning_density_per_100km2 !=
                  null
                  ? `${hazard.lightning_density_per_100km2.toFixed(
                      1
                    )} / 100 km²`
                  : "--"
              }
            />

            <Metric
              label="Hail probability"
              value={
                hazard
                  ? `${(
                      (hazard.hail_probability ??
                        0) *
                      100
                    ).toFixed(0)}%`
                  : "--"
              }
            />

            <Metric
              label="Cloudburst probability"
              value={
                hazard
                  ? `${(
                      (hazard.cloudburst_probability ??
                        0) *
                      100
                    ).toFixed(0)}%`
                  : "--"
              }
            />

            <Metric
              label="Downburst wind"
              value={
                hazard &&
                hazard.downburst_wind_kmh !=
                  null
                  ? `${hazard.downburst_wind_kmh.toFixed(
                      0
                    )} km/h`
                  : "--"
              }
            />

            <Metric
              label="Model confidence"
              value={
                hazard
                  ? `${(
                      (hazard.confidence ??
                        0) *
                      100
                    ).toFixed(0)}%`
                  : "--"
              }
            />

          </div>

          {/* =====================================
              DATA SOURCES
          ===================================== */}

          <div className="panel">

            <div className="eyebrow">
              DATA SOURCES
            </div>

            <Source
              name="DWR Radar"
              mode={sourceMode}
            />

            <Source
              name="INSAT-3D/3DR/3DS"
              mode={sourceMode}
            />

            <Source
              name="Lightning Network"
              mode={sourceMode}
            />
            <Source
              name="OPEN RADAR DATA"
              mode={sourceMode}
            />
            <div className="updated">

              Last stream update:{" "}

              {updated
                ? updated.toLocaleTimeString()
                : "--"}

            </div>

          </div>

        </aside>

      </main>

      {/* =========================================
          FOOTER
      ========================================= */}

      <footer>

        <span>
          StormCast India • AI
          Nowcasting
        </span>

        <span>
          ⚠ Not an operational only prototype
        </span>

      </footer>

    </div>
  );
}

/* =============================================
   METRIC COMPONENT
============================================= */

function Metric({
  label,
  value,
}) {
  return (
    <div className="metric">

      <span>
        {label}
      </span>

      <b>
        {value}
      </b>

    </div>
  );
}

/* =============================================
   SOURCE COMPONENT
============================================= */

function Source({
  name,
  mode,
}) {
  return (
    <div className="source">

      <span>

        <i />

        {name}

      </span>

      <b>
        {mode}
      </b>

    </div>
  );
}

export default App;