const map = L.map("map").setView([20, 0], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap" }).addTo(map);
let gpsLayer = L.layerGroup().addTo(map);
let routeLayer = L.layerGroup().addTo(map);
L.control.layers(null, { "Raw GPS": gpsLayer, "Predicted route": routeLayer }, { collapsed: false }).addTo(map);

const api = () => document.getElementById("api").value.replace(/\/$/, "");
const status = (t) => (document.getElementById("status").textContent = t);

async function loadTrips() {
  status("Connecting...");
  try {
    const h = await (await fetch(api() + "/health")).json();
    const r = await (await fetch(api() + "/trips")).json();
    status(`Connected. Model: ${h.model}. ${r.trips.length} evaluation trips.`);
    const box = document.getElementById("trips");
    box.innerHTML = "";
    r.trips.forEach((t) => {
      const b = document.createElement("button");
      b.textContent = t;
      b.onclick = () => runTrip(t, b);
      box.appendChild(b);
    });
  } catch (e) {
    status("Cannot reach API: " + e);
  }
}

async function runTrip(tripId, btn) {
  document.querySelectorAll("#trips button").forEach((b) => b.classList.remove("sel"));
  btn.classList.add("sel");
  status(`Running pipeline for trip ${tripId}...`);
  const t0 = performance.now();
  try {
    const res = await fetch(api() + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    draw(d);
    showMetrics(d, (performance.now() - t0) / 1000);
    status(`Trip ${tripId} done.`);
  } catch (e) {
    status("Error: " + e);
  }
}

function draw(d) {
  gpsLayer.clearLayers();
  routeLayer.clearLayers();
  d.gps_points.forEach((p) => L.circleMarker(p, { radius: 2, color: "#1f77b4", weight: 0, fillOpacity: 0.8 }).addTo(gpsLayer));
  d.predicted_route.forEach((seg) => L.polyline(seg, { color: "#d62728", weight: 4, opacity: 0.85 }).addTo(routeLayer));
  const all = d.gps_points.concat(d.predicted_route.flat());
  map.fitBounds(L.latLngBounds(all), { padding: [20, 20] });
}

function showMetrics(d, rtt) {
  const m = d.metrics || {};
  const f = (v) => (v === undefined || v === null ? "n/a" : typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(4)) : v);
  const rows = [
    ["Trip", d.trip_id],
    ["Model", d.model],
    ["GPS points", d.n_points],
    ["Observation Top-1 accuracy (HMM)", f(m.observation_top1_accuracy)],
    ["Top-1 accuracy (classifier only)", f(m.classifier_only_top1_accuracy)],
    ["Route precision", f(m.route_precision)],
    ["Route recall", f(m.route_recall)],
    ["Route F1", f(m.route_f1)],
    ["Route continuity (HMM)", f(m.route_continuity)],
    ["Route continuity (classifier)", f(m.classifier_only_continuity)],
    ["Scored points", `${f(m.scored_points)} / ${f(m.total_points)}`],
    ["Predicted route arcs", d.predicted_route.length],
    ["Server time (s)", f(d.timings.total_s)],
    ["Round trip (s)", rtt.toFixed(2)],
  ];
  document.getElementById("metrics").innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("");
}
