"""
End-to-End MAUSAM Pipeline Runner.
Coordinates Ingestion, Stage 1 Spherical GNN Tracking, Stage 2 Diffusion Downscaling,
Physics Loss validation, and MongoDB persistence.
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid

from .data_generator import SyntheticMeteorologicalDataGenerator
from .gnn_tracker import SphericalGNNAnomalyTracker
from .diffusion_downscaler import ConditionalDiffusionDownscaler
from ..database.mongo import db
from ..database.models import (
    ForecastRun,
    AnomalyTrack,
    DownscaledGrid,
    SpatialAlert,
    BoundingBox4D,
    TrajectoryWaypoint
)
from ..config import settings

logger = logging.getLogger("mausam.pipeline")

class MausamPipeline:
    def __init__(self):
        self.data_gen = SyntheticMeteorologicalDataGenerator()
        self.gnn_tracker = SphericalGNNAnomalyTracker(mesh_level=settings.SPHERICAL_MESH_LEVEL)
        self.diffusion_downscaler = ConditionalDiffusionDownscaler(num_timesteps=settings.DIFFUSION_STEPS)

    def run_full_pipeline(self, scenario_type: str = "cyclone_amphan", custom_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes end-to-end MAUSAM pipeline on simulated or real NetCDF data:
        1. Ingests 12 km EPS ensemble and climatology (or loaded NetCDF dataset)
        2. Tracks moving anomalies via Spherical GNN on icosahedral mesh
        3. Generates 4D temporal bounding box
        4. Performs 12km -> 5km generative diffusion downscaling preserving extreme amplitudes
        5. Computes pinpoint centroid and 5 km radius spatial alert
        6. Persists records to MongoDB
        """
        run_id = f"RUN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"Starting MAUSAM Pipeline Run {run_id} for scenario: {scenario_type}")

        # 1. Ingest Multivariable 4D EPS Data (3-10 Day lead times)
        if custom_data:
            nwp_data = custom_data
            lead_days = custom_data.get("lead_days", [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        else:
            lead_days = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            nwp_data = self.data_gen.generate_benchmark_scenario(scenario_type, lead_days=lead_days)

        # 2. Stage 1: Spherical GNN Tracking on Icosahedral Mesh
        detected_anomalies = self.gnn_tracker.track_forecast_trajectory(nwp_data, lead_days=lead_days)

        processed_anomalies = []
        downscaled_grids_list = []
        alerts_list = []

        for ano_dict in detected_anomalies:
            ano_id = ano_dict["anomaly_id"]
            bbox = ano_dict["bounding_box"]
            trajectory = ano_dict["trajectory"]

            # Save Anomaly Model
            anomaly_doc = {
                "id": str(uuid.uuid4()),
                "anomaly_id": ano_id,
                "run_id": run_id,
                "event_type": ano_dict["event_type"],
                "name": ano_dict["name"],
                "description": ano_dict["description"],
                "severity": ano_dict["severity"],
                "max_efi": ano_dict["max_efi"],
                "bounding_box": bbox,
                "trajectory": trajectory,
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            }
            db.anomalies.insert_one(anomaly_doc)
            processed_anomalies.append(anomaly_doc)

            # 3. Stage 2: Amplitude-Preserving Diffusion Downscaling (12 km -> 5 km)
            # Find the peak threat snapshot in trajectory
            peak_wp = max(trajectory, key=lambda x: x["efi_score"])
            peak_day = peak_wp["lead_day"]
            peak_day_idx = int(round(peak_day - 3.0))
            peak_day_idx = min(max(0, peak_day_idx), len(lead_days) - 1)

            # Crop macroscale bounding box from 12 km grid
            lat_mask = (nwp_data["lats"][:, 0] >= bbox["lat_min"]) & (nwp_data["lats"][:, 0] <= bbox["lat_max"])
            lon_mask = (nwp_data["lons"][0, :] >= bbox["lon_min"]) & (nwp_data["lons"][0, :] <= bbox["lon_max"])
            
            # Extract variable for downscaling (precip for cyclones/downpours, t2m for heatwaves & coldwaves)
            var_name = "temperature" if ano_dict["event_type"] in ["HEATWAVE", "COLD_WAVE"] else "precipitation"
            var_unit = "°C" if var_name == "temperature" else "mm/h"
            
            if var_name == "temperature":
                source_field = nwp_data["t2m"][peak_day_idx] - 273.15
            else:
                source_field = nwp_data["precip"][peak_day_idx]

            cropped_slice = source_field[lat_mask, :][:, lon_mask]
            if cropped_slice.size == 0 or cropped_slice.shape[0] < 4 or cropped_slice.shape[1] < 4:
                # Fallback crop around centroid
                c_lat, c_lon = peak_wp["lat"], peak_wp["lon"]
                lat_c_idx = np.argmin(np.abs(nwp_data["lats"][:, 0] - c_lat))
                lon_c_idx = np.argmin(np.abs(nwp_data["lons"][0, :] - c_lon))
                rad = 12
                r_min, r_max = max(0, lat_c_idx - rad), min(source_field.shape[0], lat_c_idx + rad)
                c_min, c_max = max(0, lon_c_idx - rad), min(source_field.shape[1], lon_c_idx + rad)
                cropped_slice = source_field[r_min:r_max, c_min:c_max]

            # Execute diffusion downscaling
            diff_results = self.diffusion_downscaler.downscale_anomaly_slice(
                cropped_slice, 
                target_shape=(48, 48),
                variable_type=var_name
            )

            # Persist downscaled 5km grid
            grid_id = f"GRID-{ano_id}-{int(peak_day)}"
            grid_doc = {
                "id": str(uuid.uuid4()),
                "grid_id": grid_id,
                "anomaly_id": ano_id,
                "run_id": run_id,
                "lead_day": peak_day,
                "variable": var_name,
                "unit": var_unit,
                "original_resolution_km": 12.0,
                "downscaled_resolution_km": 5.0,
                "centroid_lat": peak_wp["lat"],
                "centroid_lon": peak_wp["lon"],
                "grid_shape": diff_results["grid_shape"],
                "peak_amplitude_coarse": diff_results["peak_amplitude_coarse"],
                "peak_amplitude_downscaled": diff_results["peak_amplitude_downscaled"],
                "amplitude_gain_percent": diff_results["amplitude_gain_percent"],
                "diffusion_iterations": diff_results["diffusion_iterations"],
                "physics_loss_score": diff_results["physics_loss_score"],
                "moisture_convergence_score": diff_results["moisture_convergence_score"],
                "geostrophic_balance_score": diff_results["geostrophic_balance_score"],
                "grid_data": diff_results["downscaled_5km_grid"],
                "cnn_smoothed_grid": diff_results["cnn_smoothed_grid"],
                "created_at": datetime.utcnow().isoformat()
            }
            db.downscaled_grids.insert_one(grid_doc)
            downscaled_grids_list.append(grid_doc)

            # 4. Generate Pinpoint Coordinate & 5 km Categorized Spatial Alert
            alert_id = f"ALT-{int(datetime.utcnow().timestamp())}-{len(alerts_list)+1}"
            
            # Determine affected districts based on centroid location
            affected_districts = self._resolve_districts(peak_wp["lat"], peak_wp["lon"])
            
            alert_doc = {
                "id": str(uuid.uuid4()),
                "alert_id": alert_id,
                "anomaly_id": ano_id,
                "event_type": ano_dict["event_type"],
                "severity": ano_dict["severity"],
                "headline": f"{ano_dict['severity']} Alert: {ano_dict['name']} pinpointed at [{peak_wp['lat']}°N, {peak_wp['lon']}°E]",
                "description": (
                    f"MAUSAM Stage-2 Diffusion resolved 5 km impact zone with peak {var_name} of "
                    f"{diff_results['peak_amplitude_downscaled']} {var_unit} (vs blurred CNN {diff_results['peak_amplitude_cnn_smoothed']} {var_unit}). "
                    f"Lead time: Day {peak_day} (~{int(peak_day*24)}h)."
                ),
                "centroid_lat": peak_wp["lat"],
                "centroid_lon": peak_wp["lon"],
                "impact_radius_km": 5.0,
                "affected_districts": affected_districts,
                "lead_time_window": f"Day {bbox['lead_start_day']:.1f} to Day {bbox['lead_end_day']:.1f}",
                "ndrf_deployment_recommended": (ano_dict["severity"] == "SEVERE"),
                "action_instructions": [
                    "Evacuate coastal low-lying or inundation zones within 5 km radius.",
                    "Position NDRF water rescue teams and heavy de-watering pumps in vulnerable subgrids.",
                    "Issue localized agro-advisories for standing crop protection and harvest acceleration.",
                    "Pre-position power grid emergency restoration crews."
                ],
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            }
            db.alerts.insert_one(alert_doc)
            alerts_list.append(alert_doc)

        # 5. Save Forecast Run Record
        run_doc = {
            "id": str(uuid.uuid4()),
            "run_id": run_id,
            "model_source": "NEPS-G 12km Global Ensemble",
            "baseline_source": "ERA5 + IMDAA (30-Year Climatology)",
            "ensemble_members": 21,
            "initialized_at": datetime.utcnow().isoformat(),
            "forecast_horizon_days": 10,
            "detected_anomalies_count": len(processed_anomalies),
            "status": "COMPLETED"
        }
        db.forecast_runs.insert_one(run_doc)

        logger.info(f"MAUSAM Pipeline completed: {len(processed_anomalies)} anomalies tracked, {len(alerts_list)} alerts generated.")
        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "anomalies": processed_anomalies,
            "downscaled_grids": downscaled_grids_list,
            "alerts": alerts_list
        }

    def _resolve_districts(self, lat: float, lon: float) -> List[str]:
        """Resolves target Indian administrative districts near centroid."""
        # Odisha & West Bengal coast (Amphan trajectory)
        if 18.0 <= lat <= 23.5 and 84.0 <= lon <= 90.0:
            return ["South 24 Parganas", "North 24 Parganas", "Purba Medinipur", "Kendrapara", "Balasore", "Bhadrak"]
        # Northern Heatwave zone
        elif 26.0 <= lat <= 31.0 and 74.0 <= lon <= 80.0:
            return ["Churu", "Bikaner", "New Delhi NCR", "Gurugram", "Agra", "Mathura", "Alwar"]
        # Western Ghats / Mumbai downpour zone
        elif 17.5 <= lat <= 20.5 and 72.0 <= lon <= 74.5:
            return ["Mumbai Suburban", "Mumbai City", "Thane", "Raigad", "Ratnagiri", "Pune"]
        else:
            return ["Regional Coastal Belt Sector", "District Disaster Operations Cell"]
