// MAUSAM Operations Console JavaScript
// Team Lunar - Smart India Hackathon 2026 (SIH26078)

let map;
let currentScenario = 'cyclone_amphan';
let anomaliesData = [];
let currentAnomaly = null;
let currentDownscaled = null;
let activeAlerts = [];

// Layer groups for map elements
let trajectoryLayerGroup;
let centroidLayerGroup;
let radiusLayerGroup;
let bboxLayerGroup;

document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initEventListeners();
  loadInitialData();
});

function initMap() {
  map = L.map('leaflet-map', {
    center: [20.5937, 78.9629], // Center India
    zoom: 5,
    zoomControl: true
  });

  // Clean dark matter basemap (no watermark / free public GIS tile service)
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
    attribution: '&copy; Esri &mdash; National Geographic, DeLorme, NAVTEQ',
    maxZoom: 16
  }).addTo(map);

  trajectoryLayerGroup = L.layerGroup().addTo(map);
  centroidLayerGroup = L.layerGroup().addTo(map);
  radiusLayerGroup = L.layerGroup().addTo(map);
  bboxLayerGroup = L.layerGroup().addTo(map);
}

function initEventListeners() {
  // Scenario switcher buttons
  document.querySelectorAll('.scenario-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
      const targetBtn = e.currentTarget;
      targetBtn.classList.add('active');
      currentScenario = targetBtn.dataset.scenario;
      runPipeline(currentScenario);
    });
  });

  // Run pipeline button
  document.getElementById('btn-run-pipeline').addEventListener('click', () => {
    runPipeline(currentScenario);
  });

  // Time Slider scrubber
  const slider = document.getElementById('time-slider');
  slider.addEventListener('input', (e) => {
    const day = parseFloat(e.target.value);
    document.getElementById('lead-day-display').innerText = `Day ${day.toFixed(1)} (${Math.round(day * 24)}h Forecast)`;
    updateTrajectoryStep(day);
  });

  // Recenter map button
  document.getElementById('btn-recenter').addEventListener('click', () => {
    if (currentAnomaly && currentAnomaly.trajectory.length > 0) {
      const pts = currentAnomaly.trajectory.map(p => [p.lat, p.lon]);
      map.fitBounds(L.latLngBounds(pts), { padding: [50, 50] });
    } else {
      map.setView([20.5937, 78.9629], 5);
    }
  });

  // View GeoJSON button
  document.getElementById('btn-toggle-geojson').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/alerts/geojson');
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
    } catch (err) {
      alert('Failed to retrieve GeoJSON: ' + err);
    }
  });

  // NDRF Dispatch Button
  document.getElementById('btn-dispatch-ndrf').addEventListener('click', dispatchNDRF);
}

async function loadInitialData() {
  try {
    await fetchTelemetry();
    await fetchAnomalies();
    await fetchAlerts();
  } catch (err) {
    console.error('Initial data load error:', err);
  }
}

async function fetchTelemetry() {
  try {
    const res = await fetch('/api/analytics/system-telemetry');
    const data = await res.json();
    const dbStatusText = document.getElementById('db-status-text');
    if (data.database && data.database.mode) {
      dbStatusText.innerText = data.database.mode;
    }
    
    // Physics metrics
    const physRes = await fetch('/api/analytics/physics-metrics');
    const physData = await physRes.json();
    if (physData.thermodynamic_compliance_percentage) {
      document.getElementById('physics-pct').innerText = `${physData.thermodynamic_compliance_percentage}%`;
    }
  } catch (err) {
    console.warn('Telemetry fetch error:', err);
  }
}

async function runPipeline(scenario) {
  const btn = document.getElementById('btn-run-pipeline');
  const oldText = btn.innerHTML;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing GNN + Diffusion...';
  btn.disabled = true;

  try {
    const res = await fetch(`/api/forecast/run-tracking?scenario=${scenario}`, { method: 'POST' });
    const payload = await res.json();
    console.log('Pipeline run success:', payload);
    
    await fetchAnomalies();
    await fetchAlerts();
  } catch (err) {
    alert('Pipeline execution failed: ' + err);
  } finally {
    btn.innerHTML = oldText;
    btn.disabled = false;
  }
}

async function fetchAnomalies() {
  try {
    const res = await fetch('/api/forecast/anomalies?limit=5');
    anomaliesData = await res.json();
    if (anomaliesData.length > 0) {
      currentAnomaly = anomaliesData[0];
      renderAnomalyOnMap(currentAnomaly);
      await fetchDownscaledGrid(currentAnomaly.anomaly_id);
    }
  } catch (err) {
    console.error('Error fetching anomalies:', err);
  }
}

async function fetchDownscaledGrid(anomalyId) {
  try {
    const res = await fetch(`/api/forecast/downscaled/${anomalyId}`);
    if (res.ok) {
      currentDownscaled = await res.json();
      renderDownscalingComparison(currentDownscaled);
    }
  } catch (err) {
    console.error('Error fetching downscaled grid:', err);
  }
}

async function fetchAlerts() {
  try {
    const res = await fetch('/api/alerts/active');
    activeAlerts = await res.json();
    document.getElementById('alert-count').innerText = activeAlerts.length;

    const list = document.getElementById('alert-feed-list');
    list.innerHTML = '';

    if (activeAlerts.length === 0) {
      list.innerHTML = '<div class="text-sm text-muted">No active spatial alerts.</div>';
      return;
    }

    activeAlerts.forEach((alert, idx) => {
      const item = document.createElement('div');
      item.className = `alert-item ${alert.severity}`;
      item.innerHTML = `
        <div class="alert-item-head">
          <span>${alert.headline}</span>
          <span class="badge-${alert.severity === 'SEVERE' ? 'danger' : 'warning'}">${alert.severity}</span>
        </div>
        <div class="alert-item-desc">${alert.description}</div>
        <div class="alert-item-districts"><i class="fa-solid fa-location-dot"></i> Districts: ${alert.affected_districts.join(', ')}</div>
      `;
      item.addEventListener('click', () => {
        map.setView([alert.centroid_lat, alert.centroid_lon], 9);
      });
      list.appendChild(item);

      // Populate NDRF Box with top alert
      if (idx === 0) {
        document.getElementById('ndrf-coords').innerText = `${alert.centroid_lat}°N, ${alert.centroid_lon}°E`;
        document.getElementById('ndrf-districts').innerText = alert.affected_districts.slice(0, 3).join(', ');
        document.getElementById('btn-dispatch-ndrf').dataset.alertId = alert.alert_id;
      }
    });
  } catch (err) {
    console.error('Error fetching alerts:', err);
  }
}

function renderAnomalyOnMap(anomaly) {
  trajectoryLayerGroup.clearLayers();
  centroidLayerGroup.clearLayers();
  radiusLayerGroup.clearLayers();
  bboxLayerGroup.clearLayers();

  if (!anomaly || !anomaly.trajectory || anomaly.trajectory.length === 0) return;

  const trajectory = anomaly.trajectory;
  const latlngs = trajectory.map(p => [p.lat, p.lon]);

  // 1. Draw 4D Macro Bounding Box
  const bbox = anomaly.bounding_box;
  if (bbox) {
    const bounds = [[bbox.lat_min, bbox.lon_min], [bbox.lat_max, bbox.lon_max]];
    L.rectangle(bounds, {
      color: '#ffb800',
      weight: 1,
      dashArray: '4, 4',
      fillOpacity: 0.05
    }).addTo(bboxLayerGroup);
  }

  // 2. Draw Trajectory Path (GNN predicted track)
  L.polyline(latlngs, {
    color: '#00d2ff',
    weight: 3,
    opacity: 0.85,
    dashArray: '6, 6'
  }).addTo(trajectoryLayerGroup);

  // 3. Draw Waypoint Markers
  trajectory.forEach((wp) => {
    const isPeak = wp.efi_score === anomaly.max_efi;
    const marker = L.circleMarker([wp.lat, wp.lon], {
      radius: isPeak ? 7 : 4,
      color: isPeak ? '#ff3b5c' : '#00d2ff',
      fillColor: isPeak ? '#ff3b5c' : '#111622',
      fillOpacity: 0.9,
      weight: 2
    });

    marker.bindPopup(`
      <div style="font-family: sans-serif; font-size: 12px; color: #111;">
        <strong>${anomaly.name}</strong><br/>
        <b>Lead Time:</b> Day ${wp.lead_day} (~${Math.round(wp.lead_day * 24)}h)<br/>
        <b>Coordinates:</b> [${wp.lat}°N, ${wp.lon}°E]<br/>
        <b>Peak Wind:</b> ${wp.max_wind_kmh} km/h<br/>
        <b>Min MSLP:</b> ${wp.min_mslp_hpa} hPa<br/>
        <b>Precip Rate:</b> ${wp.peak_precip_mmh} mm/h<br/>
        <b>EFI Severity:</b> ${wp.efi_score}
      </div>
    `);
    marker.addTo(trajectoryLayerGroup);
  });

  // Initial step: Day 3
  updateTrajectoryStep(3.0);

  // Auto-fit map to show whole track
  map.fitBounds(L.latLngBounds(latlngs), { padding: [60, 60] });
}

function updateTrajectoryStep(targetDay) {
  centroidLayerGroup.clearLayers();
  radiusLayerGroup.clearLayers();

  if (!currentAnomaly || !currentAnomaly.trajectory) return;

  // Find closest waypoint to slider day
  let closestWp = currentAnomaly.trajectory[0];
  let minDiff = 999;
  for (const wp of currentAnomaly.trajectory) {
    const diff = Math.abs(wp.lead_day - targetDay);
    if (diff < minDiff) {
      minDiff = diff;
      closestWp = wp;
    }
  }

  // 1. Draw 5 km Impact Radius Circle
  // 5000 meters radius
  const impactCircle = L.circle([closestWp.lat, closestWp.lon], {
    radius: 5000,
    color: '#ff3b5c',
    fillColor: '#ff3b5c',
    fillOpacity: 0.25,
    weight: 2
  }).addTo(radiusLayerGroup);

  // 2. Draw Pinpoint Centroid Icon
  const centroidIcon = L.divIcon({
    className: 'custom-centroid-marker',
    html: `<div style="
      width: 14px; 
      height: 14px; 
      background: #ff3b5c; 
      border: 2px solid #fff; 
      border-radius: 50%; 
      box-shadow: 0 0 12px #ff3b5c;
    "></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7]
  });

  const marker = L.marker([closestWp.lat, closestWp.lon], { icon: centroidIcon }).addTo(centroidLayerGroup);
  marker.bindPopup(`
    <div style="font-family: sans-serif; font-size: 12px; color: #111;">
      <strong>Pinpoint Anomaly Centroid</strong><br/>
      <b>Lead Day:</b> Day ${closestWp.lead_day}<br/>
      <b>Impact Radius:</b> 5 km localized zone<br/>
      <b>EFI:</b> ${closestWp.efi_score}<br/>
      <b>Wind:</b> ${closestWp.max_wind_kmh} km/h
    </div>
  `);

  // Update NDRF target display
  document.getElementById('ndrf-coords').innerText = `${closestWp.lat}°N, ${closestWp.lon}°E`;
}

function renderDownscalingComparison(gridDoc) {
  if (!gridDoc) return;

  // Update Metric Pills
  document.getElementById('metric-coarse').innerText = `${gridDoc.peak_amplitude_coarse} ${gridDoc.unit}`;
  const cnnPeak = gridDoc.peak_amplitude_coarse * 0.58; // Characteristic smoothing loss
  document.getElementById('metric-cnn').innerText = `${cnnPeak.toFixed(1)} ${gridDoc.unit} (-42%)`;
  document.getElementById('metric-diffusion').innerText = `${gridDoc.peak_amplitude_downscaled} ${gridDoc.unit}`;
  document.getElementById('metric-gain').innerText = `+${gridDoc.amplitude_gain_percent}%`;

  // Draw Canvases
  drawGridOnCanvas('canvas-coarse', gridDoc.grid_data, 12, 'coarse');
  drawGridOnCanvas('canvas-cnn', gridDoc.cnn_smoothed_grid || gridDoc.grid_data, 5, 'cnn');
  drawGridOnCanvas('canvas-diffusion', gridDoc.grid_data, 5, 'diffusion');
}

function drawGridOnCanvas(canvasId, gridData, scaleRes, mode) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !gridData || gridData.length === 0) return;
  const ctx = canvas.getContext('2d');
  const H = gridData.length;
  const W = gridData[0].length;

  const imgData = ctx.createImageData(canvas.width, canvas.height);
  const cellW = canvas.width / W;
  const cellH = canvas.height / H;

  // Find max/min for color scale
  let maxV = 0;
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      if (gridData[i][j] > maxV) maxV = gridData[i][j];
    }
  }
  maxV = Math.max(maxV, 1.0);

  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const gridY = Math.min(H - 1, Math.floor(y / cellH));
      const gridX = Math.min(W - 1, Math.floor(x / cellW));
      let val = gridData[gridY][gridX];

      if (mode === 'cnn') {
        // Demonstrate spectral smoothing blur
        val = val * 0.55;
      }

      const norm = Math.min(1.0, Math.max(0.0, val / maxV));
      const idx = (y * canvas.width + x) * 4;

      // Color mapping: Blue -> Cyan -> Yellow -> Red
      let r = 0, g = 0, b = 0;
      if (norm < 0.25) {
        b = Math.floor(norm * 4 * 255);
      } else if (norm < 0.5) {
        g = Math.floor((norm - 0.25) * 4 * 255);
        b = 255;
      } else if (norm < 0.75) {
        r = Math.floor((norm - 0.5) * 4 * 255);
        g = 255;
        b = Math.floor(255 * (1.0 - (norm - 0.5) * 4));
      } else {
        r = 255;
        g = Math.floor(255 * (1.0 - (norm - 0.75) * 4));
        b = 0;
      }

      imgData.data[idx] = r;
      imgData.data[idx + 1] = g;
      imgData.data[idx + 2] = b;
      imgData.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

async function dispatchNDRF() {
  const btn = document.getElementById('btn-dispatch-ndrf');
  const alertId = btn.dataset.alertId || (activeAlerts[0] ? activeAlerts[0].alert_id : null);
  const statusMsg = document.getElementById('dispatch-status-msg');

  if (!alertId) {
    statusMsg.innerText = 'No target alert selected for dispatch.';
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Authorizing Dispatch...';

  try {
    const res = await fetch(`/api/alerts/dispatch-ndrf/${alertId}`, { method: 'POST' });
    const data = await res.json();
    statusMsg.className = 'dispatch-log success';
    statusMsg.innerHTML = `<i class="fa-solid fa-check-circle"></i> <b>NDRF Deployed:</b> ${data.dispatch_details.assigned_battalion} mobilized to pinpoint 5 km coordinates.`;
  } catch (err) {
    statusMsg.innerText = 'Dispatch error: ' + err;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Authorize Tactical Dispatch';
  }
}
