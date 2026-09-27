"""
End-to-End MAUSAM Production Pipeline Runner (SIH26078 Compliant).
Coordinates Operational Ingestion, Stage 1 Spherical GNN Tracking with Trained Checkpoint,
Stage 2 Conditional Diffusion Downscaling preserving extremes, Dynamic Impact Radius,
Strict Provenance Lineage, and MongoDB Persistence.
"""
import os
import uuid
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .data_generator import SyntheticMeteorologicalDataGenerator
from ..data_sources import neps_g_source, era5_baseline_source
from ..data_pipeline import AtmosphericIngestionPipeline, generate_provenance_record
from ..models import SphericalGNNModel, ConditionalDiffusionModel
from ..registry import policy_registry, model_registry
from ..database.mongo import db
from ..config import settings

logger = logging.getLogger("mausam.pipeline")

class MausamPipeline:
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.ingestion = AtmosphericIngestionPipeline(demo_mode=demo_mode)
        self.data_gen = SyntheticMeteorologicalDataGenerator()
        self.gnn_model = SphericalGNNModel(mesh_level=settings.SPHERICAL_MESH_LEVEL)
        self.diffusion_model = ConditionalDiffusionModel(num_timesteps=settings.DIFFUSION_STEPS)

    def run_full_pipeline(
        self,
        scenario_type: str = "operational_live",
        custom_data: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end MAUSAM pipeline:
        1. Ingests 12 km NEPS-G multi-member ensemble & ERA5 baseline (or custom dataset/demo benchmark)
        2. Tracks moving anomalies via Spherical GNN on icosahedral mesh using trained checkpoint
        3. Generates dynamic 4D spatio-temporal bounding box
        4. Performs 12km -> 5km amplitude-preserving conditional diffusion downscaling
        5. Computes probabilistic quantile fields and dynamic hazard impact radius
        6. Emits full cryptographic provenance and persists to MongoDB
        """
        run_id = f"RUN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        effective_mode = mode or ("demo" if scenario_type in ["cyclone_amphan", "north_india_heatwave", "monsoon_cloudburst", "north_india_coldwave"] else "live")
        is_demo = (effective_mode.lower() == "demo")

        logger.info(f"Starting MAUSAM Pipeline Run {run_id} | Mode: {effective_mode.upper()} | Scenario: {scenario_type}")

        # 1. Ingest 4D NWP Forecast Data
        lead_days = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        if custom_data:
            nwp_data = custom_data
            lead_days = custom_data.get("lead_days", lead_days)
            source_name = custom_data.get("source_name", "Uploaded_NetCDF_Dataset")
            cycle_id = custom_data.get("cycle_id", f"CUSTOM_{datetime.utcnow().strftime('%Y%m%d')}")
        elif not is_demo:
            # Operational Real-Data Path: Ingest NCMRWF NEPS-G 12km ensemble
            ingest_result = self.ingestion.ingest_operational_cycle(lead_days=lead_days)
            nwp_data = self.ingestion.normalizer.compute_dask_to_numpy(ingest_result["forecast_fields"])
            source_name = neps_g_source.source_id
            cycle_id = ingest_result["cycle_id"]
        else:
            # Explicit DEMO mode with benchmark scenario
            nwp_data = self.data_gen.generate_benchmark_scenario(scenario_type, lead_days=lead_days)
            source_name = f"DEMO_BENCHMARK_{scenario_type.upper()}"
            cycle_id = f"DEMO_CYCLE_{scenario_type}_{datetime.utcnow().strftime('%Y%m%d')}"

        # 2. Stage 1: Spherical GNN Tracking with Trained Checkpoint
        detected_anomalies = self.gnn_model.predict_anomalies_and_track(nwp_data, lead_days=lead_days)

        processed_anomalies = []
        downscaled_grids_list = []
        alerts_list = []

        # Generate run-level provenance
        provenance = generate_provenance_record(
            source_name=source_name,
            forecast_cycle=cycle_id,
            valid_time_range=f"Day {lead_days[0]:.1f} (T+{int(lead_days[0]*24)}h) to Day {lead_days[-1]:.1f} (T+{int(lead_days[-1]*24)}h)",
            mode="DEMO" if is_demo else "LIVE",
            model_version=self.gnn_model.model_version,
            checkpoint_sha=self.gnn_model.checkpoint_sha,
            uncertainty_level=0.12
        )

        for ano_dict in detected_anomalies:
            ano_id = ano_dict["anomaly_id"]
            bbox = ano_dict["bounding_box"]
            trajectory = ano_dict["trajectory"]

            # Save Anomaly Model with provenance
            anomaly_doc = {
                "id": str(uuid.uuid4()),
                "anomaly_id": ano_id,
                "run_id": run_id,
                "event_type": ano_dict["event_type"],
                "name": ano_dict["name"],
                "description": ano_dict["description"],
                "severity": ano_dict["severity"],
                "max_efi": ano_dict["max_efi"],
                "track_confidence": ano_dict["track_confidence"],
                "ensemble_spread_mean": ano_dict["ensemble_spread_mean"],
                "ensemble_members_count": ano_dict["ensemble_members_count"],
                "bounding_box": bbox,
                "trajectory": trajectory,
                "provenance": provenance,
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            }
            db.anomalies.insert_one(anomaly_doc)
            processed_anomalies.append(anomaly_doc)

            # 3. Stage 2: Amplitude-Preserving Conditional Diffusion Downscaling (12 km -> 5 km)
            peak_wp = max(trajectory, key=lambda x: x["efi_score"])
            peak_day = peak_wp["lead_day"]
            peak_day_idx = min(max(0, int(round(peak_day - lead_days[0]))), len(lead_days) - 1)

            # Crop macroscale bounding box from 12 km grid
            lats_grid = nwp_data["lats"]
            lons_grid = nwp_data["lons"]
            lat_arr = lats_grid[:, 0] if lats_grid.ndim == 2 else lats_grid
            lon_arr = lons_grid[0, :] if lons_grid.ndim == 2 else lons_grid

            lat_mask = (lat_arr >= bbox["lat_min"]) & (lat_arr <= bbox["lat_max"])
            lon_mask = (lon_arr >= bbox["lon_min"]) & (lon_arr <= bbox["lon_max"])

            var_name = "temperature" if ano_dict["event_type"] in ["HEATWAVE", "COLD_WAVE"] else "precipitation"
            var_unit = "°C" if var_name == "temperature" else "mm/h"

            # Check if ensemble member dimension is present
            source_raw = nwp_data["t2m"] if var_name == "temperature" else nwp_data["precip"]
            if source_raw.ndim == 4:
                # Average across members for coarse conditioning slice
                source_field = np.mean(source_raw[peak_day_idx], axis=0)
            elif source_raw.ndim == 3:
                source_field = source_raw[peak_day_idx]
            else:
                source_field = source_raw

            if var_name == "temperature" and np.nanmean(source_field) > 100.0:
                source_field = source_field - 273.15

            cropped_slice = source_field[lat_mask, :][:, lon_mask]
            if cropped_slice.size == 0 or cropped_slice.shape[0] < 4 or cropped_slice.shape[1] < 4:
                c_lat, c_lon = peak_wp["lat"], peak_wp["lon"]
                lat_c_idx = np.argmin(np.abs(lat_arr - c_lat))
                lon_c_idx = np.argmin(np.abs(lon_arr - c_lon))
                rad = 12
                r_min, r_max = max(0, lat_c_idx - rad), min(source_field.shape[0], lat_c_idx + rad)
                c_min, c_max = max(0, lon_c_idx - rad), min(source_field.shape[1], lon_c_idx + rad)
                cropped_slice = source_field[r_min:r_max, c_min:c_max]

            # Execute probabilistic conditional diffusion downscaling
            diff_results = self.diffusion_model.downscale_probabilistic_ensemble(
                cropped_slice,
                target_shape=(48, 48),
                variable_type=var_name,
                num_ensemble_realizations=8
            )

            # Persist downscaled 5km probabilistic grid
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
                "tail_score": diff_results["tail_score"],
                "computed_impact_radius_km": diff_results["computed_impact_radius_km"],
                "diffusion_iterations": diff_results["diffusion_timesteps"],
                "physics_loss_score": diff_results["physics_loss"],
                "moisture_convergence_score": diff_results["moisture_convergence_score"],
                "geostrophic_balance_score": diff_results["geostrophic_balance_score"],
                "thermodynamic_compliance_percentage": diff_results["thermodynamic_compliance_percentage"],
                "grid_data": diff_results["downscaled_5km_grid"], # p50 field
                "mean_grid": diff_results["mean_grid"],
                "p10_grid": diff_results["p10_grid"],
                "p90_grid": diff_results["p90_grid"],
                "uncertainty_grid": diff_results["uncertainty_grid"],
                "cnn_smoothed_grid": diff_results["cnn_smoothed_grid"],
                "provenance": provenance,
                "created_at": datetime.utcnow().isoformat()
            }
            db.downscaled_grids.insert_one(grid_doc)
            downscaled_grids_list.append(grid_doc)

            # 4. Generate Pinpoint Coordinates & Calibrated Dynamic Impact Spatial Alert
            alert_id = f"ALT-{int(datetime.utcnow().timestamp())}-{len(alerts_list)+1}"
            affected_districts = self._resolve_districts(peak_wp["lat"], peak_wp["lon"])
            
            # Policy-based severity evaluation (no magic numbers)
            calibrated_sev, exceedance_prob, rationale = policy_registry.evaluate_severity(
                event_type=ano_dict["event_type"],
                value=diff_results["peak_amplitude_downscaled"],
                efi=ano_dict["max_efi"],
                uncertainty=ano_dict.get("uncertainty_spread", 0.12)
            )

            dynamic_radius = diff_results["computed_impact_radius_km"]

            alert_doc = {
                "id": str(uuid.uuid4()),
                "alert_id": alert_id,
                "anomaly_id": ano_id,
                "event_type": ano_dict["event_type"],
                "severity": calibrated_sev,
                "headline": f"{calibrated_sev} Alert: {ano_dict['name']} pinpointed at [{peak_wp['lat']}°N, {peak_wp['lon']}°E]",
                "description": (
                    f"MAUSAM Stage-2 Diffusion resolved 5 km impact zone with peak {var_name} of "
                    f"{diff_results['peak_amplitude_downscaled']} {var_unit} (tail score: {diff_results['tail_score']}). "
                    f"Lead time: Day {peak_day} (~{int(peak_day*24)}h). Impact radius: {dynamic_radius} km. {rationale}"
                ),
                "centroid_lat": peak_wp["lat"],
                "centroid_lon": peak_wp["lon"],
                "impact_radius_km": dynamic_radius,
                "affected_districts": affected_districts,
                "lead_time_window": f"Day {bbox['lead_start_day']:.1f} to Day {bbox['lead_end_day']:.1f}",
                "exceedance_probability": exceedance_prob,
                "ndrf_deployment_recommended": (calibrated_sev == "SEVERE"),
                "action_instructions": [
                    f"Evacuate vulnerable or inundation sectors within {dynamic_radius} km impact zone.",
                    "Position NDRF water rescue teams and heavy de-watering pumps in vulnerable subgrids.",
                    "Issue localized agro-advisories for standing crop protection and harvest acceleration.",
                    "Pre-position power grid emergency restoration crews."
                ],
                "provenance": provenance,
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            }
            db.alerts.insert_one(alert_doc)
            alerts_list.append(alert_doc)

        # 5. Persist Forecast Run Record
        run_doc = {
            "id": str(uuid.uuid4()),
            "run_id": run_id,
            "forecast_cycle": cycle_id,
            "model_source": source_name,
            "baseline_source": era5_baseline_source.baseline_version,
            "ensemble_members": nwp_data.get("ensemble_members", 21),
            "initialized_at": datetime.utcnow().isoformat(),
            "forecast_horizon_days": 10,
            "detected_anomalies_count": len(processed_anomalies),
            "mode": "DEMO" if is_demo else "LIVE",
            "provenance": provenance,
            "status": "COMPLETED"
        }
        db.forecast_runs.insert_one(run_doc)

        logger.info(f"MAUSAM Pipeline Run {run_id} completed: {len(processed_anomalies)} anomalies tracked, {len(alerts_list)} alerts generated.")
        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "mode": "DEMO" if is_demo else "LIVE",
            "forecast_cycle": cycle_id,
            "source": source_name,
            "provenance": provenance,
            "anomalies": processed_anomalies,
            "downscaled_grids": downscaled_grids_list,
            "alerts": alerts_list
        }

    def _resolve_districts(self, lat: float, lon: float) -> List[str]:
        """Resolves target Indian administrative districts near centroid."""
        if 18.0 <= lat <= 23.5 and 84.0 <= lon <= 90.0:
            return ["South 24 Parganas", "North 24 Parganas", "Purba Medinipur", "Kendrapara", "Balasore", "Bhadrak"]
        elif 26.0 <= lat <= 31.0 and 74.0 <= lon <= 80.0:
            return ["Churu", "Bikaner", "New Delhi NCR", "Gurugram", "Agra", "Mathura", "Alwar"]
        elif 17.5 <= lat <= 20.5 and 72.0 <= lon <= 74.5:
            return ["Mumbai Suburban", "Mumbai City", "Thane", "Raigad", "Ratnagiri", "Pune"]
        else:
            return ["Regional Coastal Belt Sector", "District Disaster Operations Cell"]
