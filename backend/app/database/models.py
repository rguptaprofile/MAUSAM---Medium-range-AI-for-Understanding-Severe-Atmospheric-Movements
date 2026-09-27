"""
Pydantic schemas and MongoDB document representations for MAUSAM.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

class TrajectoryWaypoint(BaseModel):
    step_id: int
    lead_day: float # 3.0 to 10.0 days
    valid_time: datetime
    lat: float
    lon: float
    max_wind_kmh: float
    min_mslp_hpa: float
    peak_precip_mmh: float
    temperature_c: float
    efi_score: float = Field(ge=0.0, le=1.0)
    uncertainty_spread: float

class BoundingBox4D(BaseModel):
    lead_start_day: float = 3.0
    lead_end_day: float = 10.0
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    pressure_levels_hpa: List[int] = [1000, 850, 700, 500, 300, 200]

class AnomalyTrack(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_id: str
    run_id: str
    event_type: str # CYCLONE, HEATWAVE, EXTREME_PRECIPITATION, COLD_WAVE
    name: str
    description: str
    severity: str # LOW, MODERATE, SEVERE
    max_efi: float
    bounding_box: BoundingBox4D
    trajectory: List[TrajectoryWaypoint]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True

class DownscaledGrid(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    grid_id: str
    anomaly_id: str
    run_id: str
    lead_day: float
    variable: str # precipitation, wind_speed, temperature, mslp
    unit: str
    original_resolution_km: float = 12.0
    downscaled_resolution_km: float = 5.0
    centroid_lat: float
    centroid_lon: float
    grid_shape: List[int] # [H, W]
    peak_amplitude_coarse: float
    peak_amplitude_downscaled: float
    amplitude_gain_percent: float
    diffusion_iterations: int = 20
    physics_loss_score: float
    moisture_convergence_score: float
    geostrophic_balance_score: float
    grid_data: Optional[List[List[float]]] = None # 2D downscaled values
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SpatialAlert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str
    anomaly_id: str
    event_type: str
    severity: str # LOW, MODERATE, SEVERE
    headline: str
    description: str
    centroid_lat: float
    centroid_lon: float
    impact_radius_km: float = 5.0
    affected_districts: List[str]
    lead_time_window: str # e.g. "Day 3 (72h) - Day 5 (120h)"
    ndrf_deployment_recommended: bool
    action_instructions: List[str]
    geojson_polygon: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True

class ForecastRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    model_source: str = "NEPS-G 12km Global Ensemble"
    baseline_source: str = "ERA5 + IMDAA (30-Year Climatology)"
    ensemble_members: int = 21
    initialized_at: datetime = Field(default_factory=datetime.utcnow)
    forecast_horizon_days: int = 10
    detected_anomalies_count: int = 0
    status: str = "COMPLETED"

class DistrictSubscription(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    district_name: str
    state: str
    authority: str # NDRF, SDRF, District Collectorate, KVK (Krishi Vigyan Kendra)
    contact_email: str
    phone: str
    lat: float
    lon: float
    alert_threshold: str = "MODERATE" # LOW, MODERATE, SEVERE
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ProvenanceMetadata(BaseModel):
    model_version: str = "v1.0.0-sih26078"
    checkpoint_sha: str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    data_version: str = "NEPS-G-2026.09"
    baseline_version: str = "ERA5-IMDAA-30YR-CLIM-v1"
    code_commit: str = "git-sih26078-prod"
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class DataSourceCard(BaseModel):
    source_name: str
    provider: str
    forecast_cycle: str
    init_time: datetime
    valid_time_range: str
    file_id: str
    checksum: str
    retrieval_time: datetime = Field(default_factory=datetime.utcnow)
    quality_status: str = "VALIDATED" # VALIDATED, WARNING, DEGRADED, REJECTED
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelRegistryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_version: str
    model_type: str # GNN_TRACKER, DIFFUSION_DOWNSCALER, ENSEMBLE_PIPELINE
    checkpoint_path: str
    checkpoint_sha: str
    training_date: datetime = Field(default_factory=datetime.utcnow)
    trained_on_samples_count: int = 0
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict) # csi, crps, extreme_quantile_bias, physics_loss
    is_active: bool = False
    is_candidate: bool = False
    shadow_mode: bool = False
    status: str = "ACTIVE" # ACTIVE, CANDIDATE, ARCHIVED

class TrainingSample(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_id: str
    forecast_cycle: str
    lead_time_days: float
    init_time: datetime
    valid_time: datetime
    variable: str
    forecast_source: str # NEPS-G, NCUM-G
    verifying_truth_source: str # IMDAA, IMD_API, ERA5
    status: str = "QUEUED" # QUEUED, VERIFIED, CONSUMED_IN_TRAINING
    forecast_summary: Dict[str, Any] = Field(default_factory=dict)
    truth_summary: Dict[str, Any] = Field(default_factory=dict)
    skill_scores: Dict[str, float] = Field(default_factory=dict) # rmse, bias, csi
    created_at: datetime = Field(default_factory=datetime.utcnow)

class VerificationRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str
    forecast_cycle: str
    lead_day: float
    event_type: str # CYCLONE, HEATWAVE, DELUGE, COLDWAVE
    variable: str
    csi: float # Critical Success Index
    pod: float # Probability of Detection
    far: float # False Alarm Rate
    crps: float # Continuous Ranked Probability Score
    extreme_quantile_bias: float
    verified_at: datetime = Field(default_factory=datetime.utcnow)

