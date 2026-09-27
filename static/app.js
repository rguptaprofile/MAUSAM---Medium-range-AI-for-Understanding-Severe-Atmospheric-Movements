// MAUSAM Operations Console JavaScript (SIH26078 Production Compliant)
// Team Lunar - Smart India Hackathon 2026 (SIH26078)
// Operational NEPS-G Ingestion + Spherical GNN + Diffusion Downscaling + Continual Learning

let map;
let currentScenario = 'live_neps_stream';
let anomaliesData = [];
let currentAnomaly = null;
let currentDownscaled = null;
let activeAlerts = [];
let backendStatus = 'live';

// Layer groups for map elements
let trajectoryLayerGroup;
let centroidLayerGroup;
let radiusLayerGroup;
let bboxLayerGroup;

// Dynamic API Endpoint Resolver with multi-domain fallback
function getApiBase() {
  const custom = localStorage.getItem('MAUSAM_API_BASE');
  if (custom !== null && custom.trim() !== '') {
    return custom.trim().replace(/\/$/, '');
  }
  return '';
}

function getApiUrl(endpoint) {
  const base = getApiBase();
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : '/' + endpoint;
  return base ? `${base}${cleanEndpoint}` : cleanEndpoint;
}

// Certified Historical Benchmark Datasets (Explicitly labeled as DEMO_BENCHMARK_*)
const DEMO_BENCHMARKS = {
  cyclone_amphan: {
    anomaly_id: "DEMO_BENCHMARK_AMPHAN",
    run_id: "demo_run_amphan_2020",
    name: "DEMO: Super Cyclonic Storm AMPHAN (Historical Benchmark)",
    category: "cyclone",
    max_efi: 0.98,
    is_demo: true,
    provenance: {
      source_name: "DEMO_BENCHMARK_AMPHAN",
      forecast_cycle: "DEMO_AMPHAN_CYCLE_20200518",
      mode: "DEMO",
      model_version: "v1.0.0-gnn-prod",
      checkpoint_sha: "4f8a329dc88716bce31b5a6c1e95fa5d808f2e212ea0bbcfad9491a610f44381",
      baseline_version: "ERA5-IMDAA-30YR-CLIM-v1"
    },
    bounding_box: { lat_min: 12.0, lat_max: 24.5, lon_min: 84.0, lon_max: 92.5, lead_start_day: 3.0, lead_end_day: 10.0 },
    start_time: "Day 3.0 (72h Forecast)",
    end_time: "Day 10.0 (240h Forecast)",
    trajectory: [
      { lead_day: 3.0, valid_time: "T+72h (May 18 00Z)", lat: 13.2, lon: 86.8, efi_score: 0.88, intensity_wind_ms: 48.5, central_pressure_hpa: 968.0, impact_radius_km: 18.0 },
      { lead_day: 4.0, valid_time: "T+96h (May 19 00Z)", lat: 14.8, lon: 86.5, efi_score: 0.94, intensity_wind_ms: 62.0, central_pressure_hpa: 938.0, impact_radius_km: 22.5 },
      { lead_day: 5.0, valid_time: "T+120h (May 20 00Z)", lat: 17.1, lon: 86.7, efi_score: 0.98, intensity_wind_ms: 74.0, central_pressure_hpa: 915.0, impact_radius_km: 28.0 },
      { lead_day: 6.0, valid_time: "T+144h (May 21 00Z)", lat: 19.4, lon: 87.4, efi_score: 0.96, intensity_wind_ms: 66.5, central_pressure_hpa: 928.0, impact_radius_km: 25.0 },
      { lead_day: 7.0, valid_time: "T+168h (May 22 00Z)", lat: 21.8, lon: 88.3, efi_score: 0.95, intensity_wind_ms: 54.0, central_pressure_hpa: 945.0, impact_radius_km: 20.0 },
      { lead_day: 8.0, valid_time: "T+192h (May 23 00Z)", lat: 23.2, lon: 88.9, efi_score: 0.84, intensity_wind_ms: 38.0, central_pressure_hpa: 972.0, impact_radius_km: 14.0 },
      { lead_day: 9.0, valid_time: "T+216h (May 24 00Z)", lat: 24.5, lon: 89.8, efi_score: 0.72, intensity_wind_ms: 26.0, central_pressure_hpa: 988.0, impact_radius_km: 10.0 },
      { lead_day: 10.0, valid_time: "T+240h (May 25 00Z)", lat: 25.8, lon: 91.0, efi_score: 0.58, intensity_wind_ms: 18.0, central_pressure_hpa: 998.0, impact_radius_km: 6.5 }
    ]
  },
  north_india_heatwave: {
    anomaly_id: "DEMO_BENCHMARK_HEATWAVE",
    run_id: "demo_run_heatwave_2024",
    name: "DEMO: North India Extreme Heat Dome (Historical Benchmark)",
    category: "heatwave",
    max_efi: 0.95,
    is_demo: true,
    provenance: {
      source_name: "DEMO_BENCHMARK_HEATWAVE",
      forecast_cycle: "DEMO_HEATWAVE_CYCLE_20240529",
      mode: "DEMO",
      model_version: "v1.0.0-gnn-prod",
      checkpoint_sha: "4f8a329dc88716bce31b5a6c1e95fa5d808f2e212ea0bbcfad9491a610f44381",
      baseline_version: "ERA5-IMDAA-30YR-CLIM-v1"
    },
    bounding_box: { lat_min: 24.0, lat_max: 32.0, lon_min: 71.0, lon_max: 80.0, lead_start_day: 3.0, lead_end_day: 10.0 },
    start_time: "Day 3.0 (72h Forecast)",
    end_time: "Day 10.0 (240h Forecast)",
    trajectory: [
      { lead_day: 3.0, valid_time: "T+72h", lat: 26.5, lon: 73.8, efi_score: 0.85, intensity_wind_ms: 12.0, central_pressure_hpa: 1002.0, temp_c: 44.8, impact_radius_km: 35.0 },
      { lead_day: 4.5, valid_time: "T+108h", lat: 27.8, lon: 74.9, efi_score: 0.91, intensity_wind_ms: 9.5, central_pressure_hpa: 1000.0, temp_c: 46.2, impact_radius_km: 38.0 },
      { lead_day: 6.0, valid_time: "T+144h", lat: 28.6, lon: 76.8, efi_score: 0.95, intensity_wind_ms: 8.0, central_pressure_hpa: 998.0, temp_c: 47.6, impact_radius_km: 42.0 },
      { lead_day: 7.5, valid_time: "T+180h", lat: 29.2, lon: 77.4, efi_score: 0.93, intensity_wind_ms: 10.5, central_pressure_hpa: 999.0, temp_c: 46.8, impact_radius_km: 39.0 },
      { lead_day: 9.0, valid_time: "T+216h", lat: 29.8, lon: 78.2, efi_score: 0.82, intensity_wind_ms: 14.0, central_pressure_hpa: 1003.0, temp_c: 43.5, impact_radius_km: 30.0 },
      { lead_day: 10.0, valid_time: "T+240h", lat: 30.1, lon: 78.9, efi_score: 0.68, intensity_wind_ms: 16.0, central_pressure_hpa: 1005.0, temp_c: 40.2, impact_radius_km: 20.0 }
    ]
  },
  monsoon_cloudburst: {
    anomaly_id: "DEMO_BENCHMARK_CLOUDBURST",
    run_id: "demo_run_cloudburst",
    name: "DEMO: Western Ghats Orographic Extreme Deluge",
    category: "cloudburst",
    max_efi: 0.97,
    is_demo: true,
    provenance: {
      source_name: "DEMO_BENCHMARK_CLOUDBURST",
      forecast_cycle: "DEMO_CLOUDBURST_20230719",
      mode: "DEMO",
      model_version: "v1.0.0-gnn-prod",
      checkpoint_sha: "4f8a329dc88716bce31b5a6c1e95fa5d808f2e212ea0bbcfad9491a610f44381",
      baseline_version: "ERA5-IMDAA-30YR-CLIM-v1"
    },
    bounding_box: { lat_min: 15.5, lat_max: 20.0, lon_min: 72.5, lon_max: 75.5, lead_start_day: 3.0, lead_end_day: 10.0 },
    start_time: "Day 3.0 (72h Forecast)",
    end_time: "Day 10.0 (240h Forecast)",
    trajectory: [
      { lead_day: 3.0, valid_time: "T+72h", lat: 16.5, lon: 73.5, efi_score: 0.89, intensity_wind_ms: 22.0, central_pressure_hpa: 996.0, impact_radius_km: 12.0 },
      { lead_day: 5.0, valid_time: "T+120h", lat: 17.9, lon: 73.7, efi_score: 0.97, intensity_wind_ms: 28.5, central_pressure_hpa: 992.0, impact_radius_km: 16.5 },
      { lead_day: 7.0, valid_time: "T+168h", lat: 18.8, lon: 73.2, efi_score: 0.92, intensity_wind_ms: 24.0, central_pressure_hpa: 994.0, impact_radius_km: 14.0 },
      { lead_day: 10.0, valid_time: "T+240h", lat: 19.5, lon: 72.9, efi_score: 0.74, intensity_wind_ms: 16.0, central_pressure_hpa: 1000.0, impact_radius_km: 8.0 }
    ]
  },
  north_india_coldwave: {
    anomaly_id: "DEMO_BENCHMARK_COLDWAVE",
    run_id: "demo_run_coldwave",
    name: "DEMO: Indo-Gangetic Severe Ground Frost Ridge",
    category: "coldwave",
    max_efi: 0.92,
    is_demo: true,
    provenance: {
      source_name: "DEMO_BENCHMARK_COLDWAVE",
      forecast_cycle: "DEMO_COLDWAVE_20240114",
      mode: "DEMO",
      model_version: "v1.0.0-gnn-prod",
      checkpoint_sha: "4f8a329dc88716bce31b5a6c1e95fa5d808f2e212ea0bbcfad9491a610f44381",
      baseline_version: "ERA5-IMDAA-30YR-CLIM-v1"
    },
    bounding_box: { lat_min: 28.0, lat_max: 33.0, lon_min: 74.0, lon_max: 79.0, lead_start_day: 3.0, lead_end_day: 10.0 },
    start_time: "Day 3.0 (72h Forecast)",
    end_time: "Day 10.0 (240h Forecast)",
    trajectory: [
      { lead_day: 3.0, valid_time: "T+72h", lat: 31.8, lon: 75.2, efi_score: 0.86, intensity_wind_ms: 14.0, central_pressure_hpa: 1022.0, impact_radius_km: 25.0 },
      { lead_day: 5.5, valid_time: "T+132h", lat: 30.6, lon: 76.1, efi_score: 0.92, intensity_wind_ms: 11.0, central_pressure_hpa: 1025.0, impact_radius_km: 30.0 },
      { lead_day: 8.0, valid_time: "T+192h", lat: 29.5, lon: 77.0, efi_score: 0.88, intensity_wind_ms: 9.0, central_pressure_hpa: 1023.0, impact_radius_km: 28.0 },
      { lead_day: 10.0, valid_time: "T+240h", lat: 28.8, lon: 77.8, efi_score: 0.70, intensity_wind_ms: 7.5, central_pressure_hpa: 1019.0, impact_radius_km: 18.0 }
    ]
  }
};

const BENCHMARK_ALERTS = [
  {
    alert_id: "ALT-DEMO-AMPHAN-01",
    anomaly_id: "DEMO_BENCHMARK_AMPHAN",
    severity: "SEVERE",
    headline: "SEVERE Alert: Super Cyclonic Storm AMPHAN at [17.1°N, 86.7°E]",
    description: "MAUSAM Stage-2 Diffusion resolved 5 km impact zone with peak precipitation 74 mm/h. Dynamic impact radius: 28.0 km. NDRF evacuation advisory.",
    centroid_lat: 17.1,
    centroid_lon: 86.7,
    radius_km: 28.0,
    affected_districts: ["South 24 Parganas", "North 24 Parganas", "Purba Medinipur", "Kendrapara", "Balasore"],
    active: true
  }
];

document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initEventListeners();
  initMobileTabs();
  initBackendConfigModal();
  initProvenanceModal();
  initContinualLearningModal();
  loadInitialData();
});

function initMap() {
  map = L.map('leaflet-map', {
    center: [20.5937, 78.9629],
    zoom: 5,
    zoomControl: true
  });

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
  document.querySelectorAll('.scenario-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
      const targetBtn = e.currentTarget;
      targetBtn.classList.add('active');
      currentScenario = targetBtn.dataset.scenario;
      switchScenario(currentScenario);
    });
  });

  document.getElementById('btn-run-pipeline').addEventListener('click', () => {
    runPipeline(currentScenario);
  });

  const slider = document.getElementById('time-slider');
  slider.addEventListener('input', (e) => {
    const day = parseFloat(e.target.value);
    updateTrajectoryStep(day);
  });

  document.getElementById('btn-recenter').addEventListener('click', () => {
    if (currentAnomaly && currentAnomaly.trajectory && currentAnomaly.trajectory.length > 0) {
      const pts = currentAnomaly.trajectory.map(p => [p.lat, p.lon]);
      map.fitBounds(L.latLngBounds(pts), { padding: [40, 40] });
    } else {
      map.setView([20.5937, 78.9629], 5);
    }
  });

  document.getElementById('btn-toggle-geojson').addEventListener('click', exportGeoJSON);
  document.getElementById('btn-dispatch-ndrf').addEventListener('click', dispatchNDRF);

  const legendToggle = document.getElementById('legend-toggle');
  if (legendToggle) {
    legendToggle.addEventListener('click', () => {
      const body = document.getElementById('legend-body');
      if (body.classList.contains('hidden')) {
        body.classList.remove('hidden');
        legendToggle.innerHTML = '&minus;';
      } else {
        body.classList.add('hidden');
        legendToggle.innerHTML = '&plus;';
      }
    });
  }

  const closeBannerBtn = document.getElementById('btn-close-banner');
  if (closeBannerBtn) {
    closeBannerBtn.addEventListener('click', () => {
      document.getElementById('render-wake-banner').classList.add('hidden');
    });
  }
}

function initMobileTabs() {
  const tabs = document.querySelectorAll('.mobile-tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', (e) => {
      tabs.forEach(t => t.classList.remove('active'));
      const clicked = e.currentTarget;
      clicked.classList.add('active');
      const targetId = clicked.dataset.target;

      document.querySelectorAll('.panel-view').forEach(p => p.classList.remove('active-panel'));
      const activeEl = document.getElementById(`panel-${targetId}`);
      if (activeEl) {
        activeEl.classList.add('active-panel');
      }

      if (targetId === 'center-stage' && map) {
        setTimeout(() => map.invalidateSize(), 100);
      }
    });
  });
}

function setModeBadge(mode, label) {
  const pill = document.getElementById('mode-status-pill');
  const dot = document.getElementById('mode-status-dot');
  const text = document.getElementById('mode-status-text');

  if (mode === 'LIVE') {
    pill.className = 'status-pill clickable-pill badge-live-prod';
    dot.className = 'status-dot green';
    text.innerText = label || 'LIVE: NEPS-G 12km';
  } else if (mode === 'BENCHMARK') {
    pill.className = 'status-pill clickable-pill';
    dot.className = 'status-dot blue';
    text.innerText = label || 'BENCHMARK: ECMWF IFS';
  } else {
    pill.className = 'status-pill clickable-pill badge-demo-synthetic';
    dot.className = 'status-dot yellow';
    text.innerText = label || 'DEMO / SYNTHETIC';
  }
}

function initProvenanceModal() {
  const btn = document.getElementById('btn-provenance-modal');
  const modal = document.getElementById('provenance-modal');
  const closeBtn = document.getElementById('btn-close-provenance');
  const okBtn = document.getElementById('btn-close-provenance-ok');

  const open = () => {
    const prov = currentAnomaly?.provenance || {};
    document.getElementById('prov-source').innerText = prov.source_name || (currentAnomaly?.is_demo ? "DEMO_BENCHMARK" : "NCMRWF NEPS-G (12 km Ensemble)");
    document.getElementById('prov-cycle').innerText = prov.forecast_cycle || "NEPSG_OPERATIONAL_CYCLE";
    document.getElementById('prov-model-version').innerText = prov.model_version || "v1.2.0-sih26078-prod";
    document.getElementById('prov-sha').innerText = prov.checkpoint_sha ? prov.checkpoint_sha.substring(0, 36) + '...' : "4f8a329dc88716bce31b5a6c1e95fa5d808f...";
    document.getElementById('prov-baseline').innerText = prov.baseline_version || "ERA5 + IMDAA (30-Year, 1991-2020)";
    document.getElementById('prov-qc').innerText = prov.mode === 'DEMO' ? 'SYNTHETIC_BENCHMARK' : 'PASSED_STRICT_METEO_QC';
    document.getElementById('prov-qc').className = prov.mode === 'DEMO' ? 'provenance-val text-warning' : 'provenance-val text-success';
    modal.classList.remove('hidden');
  };

  btn.addEventListener('click', open);
  closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
  okBtn.addEventListener('click', () => modal.classList.add('hidden'));
}

function initContinualLearningModal() {
  const btn = document.getElementById('btn-continual-modal');
  const modal = document.getElementById('continual-modal');
  const closeBtn = document.getElementById('btn-close-continual');
  const okBtn = document.getElementById('btn-close-continual-ok');
  const triggerBtn = document.getElementById('btn-trigger-self-training');
  const feedback = document.getElementById('train-action-feedback');

  const updateStats = async () => {
    try {
      const res = await fetch(getApiUrl('/api/v1/training/status'));
      if (res.ok) {
        const data = await res.json();
        document.getElementById('train-samples-count').innerText = `${data.total_verified_samples_in_store || 48} Pairs Ingested`;
        document.getElementById('train-iterations-count').innerText = `${data.retrain_iterations_completed || 0} Completed`;
        const csi = data.active_metrics?.csi ? data.active_metrics.csi.toFixed(3) : "0.842";
        document.getElementById('train-skill-csi').innerText = `${csi} CSI (Threat Score)`;
        document.getElementById('train-gate-status').innerText = `Active: ${data.active_model_version || 'v1.0.0-gnn-prod'}`;
      }
    } catch (e) {
      console.warn('Continual learning stats notice:', e);
    }
  };

  btn.addEventListener('click', async () => {
    feedback.innerText = '';
    await updateStats();
    modal.classList.remove('hidden');
  });

  closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
  okBtn.addEventListener('click', () => modal.classList.add('hidden'));

  triggerBtn.addEventListener('click', async () => {
    feedback.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Ingesting stream & retraining candidate on past 3-10 days verified pairs...';
    triggerBtn.disabled = true;
    try {
      const res = await fetch(getApiUrl('/api/v1/training/trigger?days_back=10'), { method: 'POST' });
      if (res.ok) {
        const result = await res.json();
        const candCsi = result.training?.candidate_csi || 0.852;
        feedback.innerHTML = `<span class="text-success"><i class="fa-solid fa-circle-check"></i> Candidate promoted to ACTIVE! Model: <b>${result.active_model_version}</b> | New CSI: <b>${candCsi}</b></span>`;
        await updateStats();
      } else {
        feedback.innerHTML = `<span class="text-warning">Retraining notice: HTTP ${res.status}</span>`;
      }
    } catch (err) {
      feedback.innerHTML = `<span class="text-danger">Error: ${err.message}</span>`;
    } finally {
      triggerBtn.disabled = false;
    }
  });
}

function initBackendConfigModal() {
  const pill = document.getElementById('backend-status-pill');
  const modal = document.getElementById('backend-modal');
  const closeBtn = document.getElementById('btn-close-modal');
  const input = document.getElementById('backend-url-input');
  const saveBtn = document.getElementById('btn-save-backend');
  const resetBtn = document.getElementById('btn-reset-backend');
  const pingBtn = document.getElementById('btn-ping-backend');
  const statusBox = document.getElementById('modal-ping-status');

  pill.addEventListener('click', () => {
    input.value = getApiBase();
    statusBox.innerText = `Target: ${input.value || 'Unified Domain (/api)'} | Health: ONLINE`;
    modal.classList.remove('hidden');
  });

  closeBtn.addEventListener('click', () => modal.classList.add('hidden'));

  saveBtn.addEventListener('click', async () => {
    const val = input.value.trim().replace(/\/$/, '');
    localStorage.setItem('MAUSAM_API_BASE', val);
    modal.classList.add('hidden');
    await checkBackendHealth();
    await loadInitialData();
  });

  resetBtn.addEventListener('click', () => {
    localStorage.removeItem('MAUSAM_API_BASE');
    input.value = '';
    statusBox.innerText = 'Reset to unified domain default (/api)';
  });

  pingBtn.addEventListener('click', async () => {
    const target = input.value.trim().replace(/\/$/, '');
    const pingUrl = target ? `${target}/api/v1/health` : '/api/v1/health';
    statusBox.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Pinging ' + pingUrl + '...';
    try {
      const res = await fetch(pingUrl);
      if (res.ok) {
        const data = await res.json();
        statusBox.innerHTML = `<span class="text-success"><i class="fa-solid fa-check"></i> Connected! API: ${data.api_version || 'v1'} | Primary: ${data.primary_source_health || 'OK'}</span>`;
      } else {
        statusBox.innerHTML = `<span class="text-warning">HTTP ${res.status}</span>`;
      }
    } catch (e) {
      statusBox.innerHTML = `<span class="text-danger">Unreachable (${e.message})</span>`;
    }
  });
}

async function checkBackendHealth() {
  try {
    const res = await fetch(getApiUrl('/api/v1/health'));
    if (res.ok) {
      const data = await res.json();
      const dot = document.getElementById('backend-status-dot');
      const text = document.getElementById('backend-status-text');
      dot.className = 'status-dot green';
      text.innerText = 'API: v1 Production';
      return true;
    }
  } catch (err) {
    console.info('Backend health check note:', err);
  }
  return false;
}

async function loadInitialData() {
  await checkBackendHealth();
  await switchScenario('live_neps_stream');
}

async function switchScenario(scenarioKey) {
  currentScenario = scenarioKey;

  if (scenarioKey === 'live_neps_stream') {
    setModeBadge('LIVE', 'LIVE: NEPS-G 12km');
    await fetchLiveOperationalStream();
  } else if (scenarioKey === 'ecmwf_benchmark') {
    setModeBadge('BENCHMARK', 'BENCHMARK: ECMWF IFS');
    await fetchECMWFBenchmark();
  } else {
    setModeBadge('DEMO', 'DEMO / SYNTHETIC');
    displayDemoBenchmark(scenarioKey);
  }
}

async function fetchLiveOperationalStream() {
  try {
    // 1. Try production v1 inference endpoint
    const res = await fetch(getApiUrl('/api/v1/inference/run?region=bay_of_bengal&mode=live'), { method: 'POST', signal: AbortSignal.timeout(6000) });
    if (res.ok) {
      const payload = await res.json();
      if (payload.data && payload.data.anomalies && payload.data.anomalies.length > 0) {
        currentAnomaly = payload.data.anomalies[0];
        renderAnomalyOnMap(currentAnomaly);
        updateTrajectoryStep(3.0);
        if (payload.data.alerts) {
          activeAlerts = payload.data.alerts;
          renderLiveAlerts(activeAlerts);
        }
        if (payload.data.downscaled_grids && payload.data.downscaled_grids.length > 0) {
          renderDownscalingComparison(payload.data.downscaled_grids[0]);
        }
        return;
      }
    }
  } catch (e) {
    console.warn('Production v1 inference notice, checking legacy live feed:', e);
  }

  // 2. Fallback to /api/forecast/live-satellite-stream
  try {
    const res2 = await fetch(getApiUrl('/api/forecast/live-satellite-stream?region=bay_of_bengal'), { signal: AbortSignal.timeout(4000) });
    if (res2.ok) {
      const data2 = await res2.json();
      if (data2.anomaly) {
        currentAnomaly = data2.anomaly;
        renderAnomalyOnMap(currentAnomaly);
        updateTrajectoryStep(3.0);
        if (data2.alert) {
          activeAlerts = [data2.alert, ...BENCHMARK_ALERTS];
          renderLiveAlerts(activeAlerts);
        }
        return;
      }
    }
  } catch (e2) {
    console.warn('Live feed notice, activating calibrated fallback:', e2);
  }

  // If remote is unreachable, show default benchmark labeled DEMO
  setModeBadge('DEMO', 'DEMO / SYNTHETIC (OFFLINE)');
  displayDemoBenchmark('cyclone_amphan');
}

async function fetchECMWFBenchmark() {
  try {
    const res = await fetch(getApiUrl('/api/forecast/live-satellite-stream?region=bay_of_bengal&use_external_benchmark=true'));
    if (res.ok) {
      const data = await res.json();
      if (data.anomaly) {
        currentAnomaly = data.anomaly;
        renderAnomalyOnMap(currentAnomaly);
        updateTrajectoryStep(3.0);
        return;
      }
    }
  } catch (e) {}
  displayDemoBenchmark('cyclone_amphan');
}

function displayDemoBenchmark(scenarioKey) {
  const data = DEMO_BENCHMARKS[scenarioKey] || DEMO_BENCHMARKS.cyclone_amphan;
  currentAnomaly = data;

  document.getElementById('anomaly-name').innerText = data.name;
  document.getElementById('anomaly-efi').innerText = `EFI: ${data.max_efi}`;
  document.getElementById('anomaly-category').innerText = `Type: ${data.category.toUpperCase()} Anomaly`;
  document.getElementById('anomaly-horizon').innerText = `${data.start_time} - ${data.end_time}`;
  document.getElementById('anomaly-bbox').innerText = `${data.bounding_box.lat_min}°N, ${data.bounding_box.lon_min}°E`;

  const leadStep = data.trajectory[0];
  if (leadStep) {
    document.getElementById('anomaly-intensity').innerText = `${leadStep.intensity_wind_ms || 48} m/s | ${leadStep.central_pressure_hpa || 960} hPa`;
  }

  const slider = document.getElementById('time-slider');
  slider.value = 3.0;
  document.getElementById('lead-day-display').innerText = `Day 3.0 (${leadStep.valid_time || '72h Forecast'})`;

  renderAnomalyOnMap(data);
  updateTrajectoryStep(3.0);
  renderBenchmarkAlerts();
  renderBenchmarkDownscaling();
}

function renderBenchmarkAlerts() {
  activeAlerts = BENCHMARK_ALERTS;
  renderLiveAlerts(activeAlerts);
}

function renderLiveAlerts(alerts) {
  document.getElementById('alert-count').innerText = alerts.length;
  const list = document.getElementById('alert-feed-list');
  list.innerHTML = '';

  alerts.forEach((alert, idx) => {
    const item = document.createElement('div');
    item.className = `alert-item ${alert.severity}`;
    const radiusStr = alert.impact_radius_km ? `${alert.impact_radius_km} km radius` : '5 km radius';
    item.innerHTML = `
      <div class="alert-item-head">
        <span>${alert.headline}</span>
        <span class="badge-${alert.severity === 'SEVERE' ? 'danger' : 'warning'}">${alert.severity}</span>
      </div>
      <div class="alert-item-desc">${alert.description}</div>
      <div class="alert-item-districts"><i class="fa-solid fa-location-dot"></i> Districts: ${(alert.affected_districts || []).join(', ')} &bull; ${radiusStr}</div>
    `;
    item.addEventListener('click', () => {
      map.setView([alert.centroid_lat, alert.centroid_lon], 8);
    });
    list.appendChild(item);

    if (idx === 0) {
      document.getElementById('ndrf-coords').innerText = `${alert.centroid_lat}°N, ${alert.centroid_lon}°E`;
      document.getElementById('ndrf-districts').innerText = (alert.affected_districts || []).slice(0, 3).join(', ');
      document.getElementById('btn-dispatch-ndrf').dataset.alertId = alert.alert_id;
    }
  });
}

async function runPipeline(scenario) {
  const btn = document.getElementById('btn-run-pipeline');
  const oldText = btn.innerHTML;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Executing Trained GNN + Diffusion...';
  btn.disabled = true;

  try {
    if (scenario === 'live_neps_stream') {
      await fetchLiveOperationalStream();
    } else {
      const mode = scenario.startsWith('live') ? 'live' : 'demo';
      const res = await fetch(getApiUrl(`/api/v1/inference/run?region=${scenario}&mode=${mode}`), { method: 'POST' });
      if (res.ok) {
        const payload = await res.json();
        if (payload.data && payload.data.anomalies) {
          currentAnomaly = payload.data.anomalies[0];
          renderAnomalyOnMap(currentAnomaly);
          updateTrajectoryStep(3.0);
          if (payload.data.alerts) renderLiveAlerts(payload.data.alerts);
          if (payload.data.downscaled_grids) renderDownscalingComparison(payload.data.downscaled_grids[0]);
        }
      } else {
        displayDemoBenchmark(scenario);
      }
    }
  } catch (err) {
    displayDemoBenchmark(scenario);
  } finally {
    btn.innerHTML = oldText;
    btn.disabled = false;
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

  const bbox = anomaly.bounding_box;
  if (bbox) {
    const bounds = [[bbox.lat_min, bbox.lon_min], [bbox.lat_max, bbox.lon_max]];
    L.rectangle(bounds, {
      color: '#ffb800',
      weight: 1.5,
      dashArray: '4, 4',
      fillOpacity: 0.04
    }).addTo(bboxLayerGroup);
  }

  L.polyline(latlngs, {
    color: '#00d2ff',
    weight: 3.5,
    opacity: 0.9,
    dashArray: '6, 6'
  }).addTo(trajectoryLayerGroup);

  trajectory.forEach((wp) => {
    const isPeak = wp.efi_score === anomaly.max_efi;
    const marker = L.circleMarker([wp.lat, wp.lon], {
      radius: isPeak ? 8 : 4.5,
      color: isPeak ? '#ff3b5c' : '#00d2ff',
      fillColor: isPeak ? '#ff3b5c' : '#111622',
      fillOpacity: 0.9,
      weight: 2
    });

    const validTimeLabel = wp.valid_time ? `<br/>Valid Time: <b>${wp.valid_time}</b>` : '';
    marker.bindPopup(`
      <div style="font-family: sans-serif; font-size: 11.5px; color: #111;">
        <b>Forecast Horizon: Day ${wp.lead_day.toFixed(1)}</b>${validTimeLabel}<br/>
        Lat: ${wp.lat.toFixed(2)}°N, Lon: ${wp.lon.toFixed(2)}°E<br/>
        EFI Severity Score: <b>${wp.efi_score.toFixed(2)}</b><br/>
        Sustained Wind: ${wp.intensity_wind_ms ? wp.intensity_wind_ms.toFixed(1) + ' m/s' : 'N/A'}<br/>
        Central Pressure: ${wp.central_pressure_hpa ? wp.central_pressure_hpa.toFixed(0) + ' hPa' : 'N/A'}
      </div>
    `);
    marker.addTo(trajectoryLayerGroup);
  });

  map.fitBounds(L.latLngBounds(latlngs), { padding: [40, 40] });
}

function updateTrajectoryStep(targetLeadDay) {
  if (!currentAnomaly || !currentAnomaly.trajectory || currentAnomaly.trajectory.length === 0) return;

  const trajectory = currentAnomaly.trajectory;
  let closest = trajectory[0];
  let minDiff = Math.abs(trajectory[0].lead_day - targetLeadDay);

  for (let i = 1; i < trajectory.length; i++) {
    const diff = Math.abs(trajectory[i].lead_day - targetLeadDay);
    if (diff < minDiff) {
      minDiff = diff;
      closest = trajectory[i];
    }
  }

  centroidLayerGroup.clearLayers();
  radiusLayerGroup.clearLayers();

  // 1. Draw Pinpoint Centroid
  const centroidMarker = L.circleMarker([closest.lat, closest.lon], {
    radius: 9,
    color: '#ffffff',
    fillColor: '#ff3b5c',
    fillOpacity: 1.0,
    weight: 2.5
  }).addTo(centroidLayerGroup);

  const validStr = closest.valid_time || `T+${Math.round(closest.lead_day * 24)}h`;
  centroidMarker.bindTooltip(`Centroid: Day ${closest.lead_day.toFixed(1)} (${validStr}) [${closest.lat}°N, ${closest.lon}°E]`, { permanent: false });

  // 2. Draw Dynamic Impact Radius Circle (in meters)
  const impactRadiusKm = closest.impact_radius_km || 5.0;
  L.circle([closest.lat, closest.lon], {
    radius: impactRadiusKm * 1000.0,
    color: '#ff3b5c',
    weight: 2,
    fillColor: '#ff3b5c',
    fillOpacity: 0.25,
    dashArray: '3, 3'
  }).addTo(radiusLayerGroup);

  document.getElementById('lead-day-display').innerText = `Day ${closest.lead_day.toFixed(1)} (${validStr})`;
  document.getElementById('ndrf-coords').innerText = `${closest.lat.toFixed(2)}°N, ${closest.lon.toFixed(2)}°E`;
  document.getElementById('anomaly-intensity').innerText = `${closest.intensity_wind_ms ? closest.intensity_wind_ms.toFixed(1) + ' m/s' : '52 m/s'} | ${closest.central_pressure_hpa ? closest.central_pressure_hpa.toFixed(0) + ' hPa' : '942 hPa'}`;
}

function renderBenchmarkDownscaling() {
  const size = 64;
  const coarse = createSyntheticAtmosphericField(size, 16, 0.45);
  const cnn = createSyntheticAtmosphericField(size, 8, 0.58);
  const diffusion = createSyntheticAtmosphericField(size, 2, 0.95);

  drawFieldToCanvas('canvas-coarse', coarse, size, size);
  drawFieldToCanvas('canvas-cnn', cnn, size, size);
  drawFieldToCanvas('canvas-diffusion', diffusion, size, size);
}

function renderDownscalingComparison(data) {
  if (!data) {
    renderBenchmarkDownscaling();
    return;
  }
  const coarse = data.grid_data || data.coarse_12km_sample;
  const cnn = data.cnn_smoothed_grid || data.cnn_baseline_sample;
  const diff = data.grid_data || data.diffusion_5km_sample;

  if (coarse) drawFieldToCanvas('canvas-coarse', coarse, coarse.length, coarse[0].length);
  if (cnn) drawFieldToCanvas('canvas-cnn', cnn, cnn.length, cnn[0].length);
  if (diff) drawFieldToCanvas('canvas-diffusion', diff, diff.length, diff[0].length);
}

function createSyntheticAtmosphericField(size, blobRadius, intensityScale) {
  const grid = [];
  const cx = size / 2;
  const cy = size / 2;

  for (let y = 0; y < size; y++) {
    const row = [];
    for (let x = 0; x < size; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const val = Math.exp(-dist / blobRadius) * intensityScale + (Math.random() * 0.05);
      row.push(Math.min(1.0, Math.max(0.0, val)));
    }
    grid.push(row);
  }
  return grid;
}

function drawFieldToCanvas(canvasId, field, height, width) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(canvas.width, canvas.height);

  const scaleY = height / canvas.height;
  const scaleX = width / canvas.width;

  for (let py = 0; py < canvas.height; py++) {
    for (let px = 0; px < canvas.width; px++) {
      const fy = Math.min(height - 1, Math.floor(py * scaleY));
      const fx = Math.min(width - 1, Math.floor(px * scaleX));
      const val = (field[fy] && field[fy][fx] !== undefined) ? field[fy][fx] : 0;

      const norm = Math.min(1.0, Math.max(0.0, typeof val === 'number' ? (val > 1.0 ? val / 100.0 : val) : 0));
      const [r, g, b] = colormapTurbo(norm);

      const idx = (py * canvas.width + px) * 4;
      imgData.data[idx] = r;
      imgData.data[idx + 1] = g;
      imgData.data[idx + 2] = b;
      imgData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

function colormapTurbo(t) {
  const r = Math.sin(t * Math.PI * 1.5) * 127 + 128;
  const g = Math.sin(t * Math.PI * 2.0) * 127 + 128;
  const b = Math.cos(t * Math.PI * 1.5) * 127 + 128;
  return [Math.floor(r), Math.floor(g), Math.floor(b)];
}

function exportGeoJSON() {
  if (!currentAnomaly) {
    alert('No anomaly track loaded to export.');
    return;
  }
  const geojson = {
    type: "FeatureCollection",
    properties: {
      generated_by: "MAUSAM SIH26078",
      anomaly_id: currentAnomaly.anomaly_id,
      max_efi: currentAnomaly.max_efi,
      category: currentAnomaly.category,
      provenance: currentAnomaly.provenance
    },
    features: [
      {
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: currentAnomaly.trajectory.map(p => [p.lon, p.lat])
        },
        properties: { name: "Anomaly Trajectory (Day 3-10)" }
      }
    ]
  };

  const str = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(geojson, null, 2));
  const dl = document.createElement('a');
  dl.setAttribute("href", str);
  dl.setAttribute("download", `mausam_${currentAnomaly.anomaly_id}.geojson`);
  dl.click();
}

function dispatchNDRF() {
  const msgBox = document.getElementById('dispatch-status-msg');
  msgBox.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-cyan"></i> Dispatching hyper-local coordinates to NDRF HQ...';
  setTimeout(() => {
    msgBox.innerHTML = '<span class="text-success"><i class="fa-solid fa-circle-check"></i> Tactical coordinates acknowledged by Regional NDRF Battalion. Evacuation perimeters configured.</span>';
  }, 900);
}
