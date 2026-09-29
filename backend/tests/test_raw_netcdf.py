"""
Tests for Native NetCDF Ingestion, Continual Training & Pipeline Execution (data/raw.nc).
Validates SIH26078 compliance for real atmospheric datasets.
"""
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.data_sources import raw_netcdf_source
from backend.app.training import CanonicalDatasetBuilder, continual_learning_engine
from backend.app.core.pipeline import MausamPipeline

client = TestClient(app)

class TestRawNetCDFIntegration:
    @pytest.fixture(autouse=True)
    def check_dataset_exists(self):
        assert os.path.exists("data/raw.nc"), "data/raw.nc must exist for this test suite"

    def test_raw_netcdf_connector_metadata(self):
        """Validates that raw.nc structural metadata and temperature bounds are correctly parsed."""
        meta = raw_netcdf_source.inspect_dataset_metadata()
        assert "dimensions" in meta
        assert meta["dimensions"]["latitude"] == 61
        assert meta["dimensions"]["longitude"] == 61
        assert meta["dimensions"]["valid_time"] == 288
        
        t_stats = meta["temperature_stats"]
        assert t_stats["max_celsius"] >= 50.0 # Historic North India heatwave exceeded 50°C
        assert t_stats["min_celsius"] < 0.0

    def test_raw_netcdf_fetch_cycle(self):
        """Tests medium-range 4D cycle extraction across Days 3 to 10."""
        lead_days = [3.0, 5.0, 7.0, 10.0]
        cycle = raw_netcdf_source.fetch_cycle(cycle_id="RAW_NC_TEST", lead_days=lead_days)
        
        assert cycle["source_name"] == "ECMWF_REAL_NETCDF_RAW"
        assert cycle["t2m"].shape == (4, 21, 61, 61) # 4 leads, 21 ensemble members
        assert cycle["mslp"].shape == (4, 21, 61, 61)
        assert cycle["u_wind"].shape == (4, 21, 61, 61)
        assert cycle["v_wind"].shape == (4, 21, 61, 61)
        
        # Verify physical realism of thermal depression: MSLP in hot core reaches sub-1000 hPa
        min_p = float(np.min(cycle["mslp"]))
        assert 990.0 <= min_p <= 1005.0

    def test_canonical_dataset_builder_from_raw_nc(self):
        """Tests verified ground-truth training sample generation from raw.nc."""
        builder = CanonicalDatasetBuilder()
        samples = builder.build_samples_from_raw_nc("data/raw.nc")
        assert isinstance(samples, list)
        if len(samples) > 0:
            s0 = samples[0]
            assert "sample_id" in s0
            assert "skill_scores" in s0
            assert s0["skill_scores"]["csi"] >= 0.85

    def test_continual_learning_ingest_and_train(self):
        """Tests full continual self-training cycle on data/raw.nc."""
        res = continual_learning_engine.ingest_raw_netcdf_dataset("data/raw.nc")
        assert res["status"] == "COMPLETED"
        assert res["source_dataset"] == "data/raw.nc"
        assert "training_result" in res
        t_res = res["training_result"]
        assert "candidate_version" in t_res
        assert "metrics" in t_res
        assert "gnn_loss" in t_res["metrics"]
        assert "diffusion_loss" in t_res["metrics"]

    def test_full_11_stage_pipeline_on_raw_netcdf(self):
        """Tests end-to-end 11-stage pipeline on raw.nc heatwave scenario."""
        pipe = MausamPipeline()
        summary = pipe.run_full_pipeline(scenario_type="raw_netcdf", mode="live")
        
        assert summary["status"] == "SUCCESS"
        assert summary["total_stages_executed"] == 11
        assert len(summary["stages"]) == 11
        
        # Anomaly classification must correctly identify HEATWAVE
        anomaly = summary["anomaly"]
        assert anomaly["event_type"] == "HEATWAVE"
        assert anomaly["severity"] in ["ORANGE", "RED"]
        assert anomaly["max_efi"] >= 0.85
        
        # Downscaling must resolve temperature field
        down = summary["downscaling"]
        assert down["variable"] == "temperature_c"
        assert down["target_resolution_km"] == 5.0
        assert down["peak_5km_diffusion"] >= 45.0

    def test_api_raw_netcdf_endpoints(self):
        """Tests FastAPI REST endpoints for data/raw.nc."""
        # 1. Metadata
        r_meta = client.get("/api/v1/raw-nc/metadata")
        assert r_meta.status_code == 200
        assert r_meta.json()["status"] == "success"
        
        # 2. Ingest and Train
        r_train = client.post("/api/v1/raw-nc/ingest-and-train")
        assert r_train.status_code == 200
        assert r_train.json()["status"] == "success"
        
        # 3. Run Pipeline
        r_pipe = client.post("/api/v1/raw-nc/run-pipeline")
        assert r_pipe.status_code == 200
        res = r_pipe.json()
        assert res["status"] == "success"
        assert res["pipeline_summary"]["total_stages_executed"] == 11
