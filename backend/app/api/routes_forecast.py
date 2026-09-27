from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from typing import Dict, Any, List, Optional
import os
import shutil
from ..core.pipeline import MausamPipeline
from ..core.real_data_loader import RealAtmosphericDataLoader
from ..database.mongo import db

router = APIRouter(prefix="/api/forecast", tags=["Forecast & Anomaly Tracking"])
pipeline = MausamPipeline()
real_loader = RealAtmosphericDataLoader()

@router.post("/run-tracking")
def trigger_tracking_pipeline(scenario: str = Query("cyclone_amphan", description="Scenario type: cyclone_amphan, north_india_heatwave, or monsoon_cloudburst")) -> Dict[str, Any]:
    """
    Executes the full MAUSAM AI tracking & downscaling pipeline:
    1. Ingestion of 12km NEPS-G 4D Ensemble & ERA5 baseline.
    2. Spherical GNN anomaly tracking on icosahedral mesh.
    3. Macroscale 4D bounding box & trajectory over 3-10 day window.
    4. Amplitude-preserving diffusion downscaling to 5km subgrid.
    5. MongoDB persistence.
    """
    valid_scenarios = ["cyclone_amphan", "north_india_heatwave", "monsoon_cloudburst", "north_india_coldwave"]
    if scenario not in valid_scenarios:
        raise HTTPException(status_code=400, detail=f"Invalid scenario. Choose from: {valid_scenarios}")
        
    result = pipeline.run_full_pipeline(scenario_type=scenario)
    return {
        "status": "success",
        "message": f"Pipeline successfully executed for scenario: {scenario}",
        "data": result
    }

@router.get("/runs")
def list_forecast_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """Lists ingested forecast runs and execution summaries."""
    return db.forecast_runs.find({}, sort=[("initialized_at", -1)], limit=limit)

@router.get("/anomalies")
def list_anomalies(
    run_id: Optional[str] = None, 
    severity: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Lists detected severe atmospheric anomalies with 4D trajectories."""
    query = {}
    if run_id:
        query["run_id"] = run_id
    if severity:
        query["severity"] = severity.upper()
    return db.anomalies.find(query, sort=[("created_at", -1)], limit=limit)

@router.get("/anomalies/{anomaly_id}")
def get_anomaly_detail(anomaly_id: str) -> Dict[str, Any]:
    """Retrieves full 4D trajectory, waypoints, and bounding box of an anomaly."""
    anomaly = db.anomalies.find_one({"anomaly_id": anomaly_id})
    if not anomaly:
        raise HTTPException(status_code=404, detail=f"Anomaly with id {anomaly_id} not found")
    return anomaly

@router.get("/downscaled/{anomaly_id}")
def get_downscaled_grid(anomaly_id: str) -> Dict[str, Any]:
    """
    Retrieves the 5km generative diffusion downscaled subgrid,
    including comparison metrics with traditional CNN smoothing.
    """
    grid = db.downscaled_grids.find_one({"anomaly_id": anomaly_id})
    if not grid:
        raise HTTPException(status_code=404, detail=f"No downscaled grid found for anomaly {anomaly_id}")
    return grid

@router.post("/upload-netcdf")
async def upload_and_process_netcdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Uploads a real meteorological NetCDF (.nc) file and executes the MAUSAM
    Spherical GNN + Generative Diffusion pipeline on real atmospheric fields.
    """
    if not file.filename.endswith((".nc", ".nc4", ".netcdf", ".grib", ".grib2")):
        raise HTTPException(status_code=400, detail="Uploaded file must be a NetCDF or GRIB file (.nc, .nc4, .grib2)")
        
    upload_dir = os.path.join(os.getcwd(), "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        real_data = real_loader.load_netcdf(temp_path)
        result = pipeline.run_full_pipeline(scenario_type="real_netcdf_ingest", custom_data=real_data)
        return {
            "status": "success",
            "message": f"Successfully ingested and processed real atmospheric file: {file.filename}",
            "filename": file.filename,
            "anomalies_detected": len(result["anomalies"]),
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process real NetCDF file: {str(e)}")

@router.post("/process-real-data")
def process_real_data_file(file_path: str = Query("data/sample_real_neps_g.nc")) -> Dict[str, Any]:
    """
    Processes a real local or downloaded NetCDF file path via Xarray.
    """
    if not os.path.exists(file_path):
        # Auto-generate sample if requested sample doesn't exist yet
        if "sample" in file_path:
            real_loader.generate_sample_real_netcdf(file_path)
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    real_data = real_loader.load_netcdf(file_path)
    result = pipeline.run_full_pipeline(scenario_type="real_nwp_dataset", custom_data=real_data)
    return {
        "status": "success",
        "message": f"Successfully processed real dataset: {file_path}",
        "anomalies_detected": len(result["anomalies"]),
        "data": result
    }

from ..core.live_nwp_service import live_nwp_service

@router.get("/live-satellite-stream")
def get_live_satellite_stream(
    region: str = Query("bay_of_bengal", description="Sector: bay_of_bengal, arabian_sea, delhi_ncr, western_ghats, odisha_coast"),
    lat: Optional[float] = Query(None, description="Custom latitude"),
    lon: Optional[float] = Query(None, description="Custom longitude"),
    use_external_benchmark: bool = Query(False, description="Whether to query external point benchmark instead of primary NEPS-G")
) -> Dict[str, Any]:
    """
    Fetches operational NWP stream (NCMRWF NEPS-G 12 km by default, or labeled external benchmark),
    computes EFI extreme deviations, tracks trajectory, and stores in MongoDB.
    """
    live_anomaly = live_nwp_service.fetch_live_ensemble(
        region_key=region, 
        custom_lat=lat, 
        custom_lon=lon,
        use_external_benchmark=use_external_benchmark
    )
    
    # Save to MongoDB
    db.anomalies.update_one(
        {"anomaly_id": live_anomaly["anomaly_id"]},
        {"$set": live_anomaly},
        upsert=True
    )
    
    # Generate live spatial alert for NDRF
    first_wp = live_anomaly["trajectory"][0]
    alert_doc = {
        "alert_id": f"alert_{live_anomaly['anomaly_id']}",
        "anomaly_id": live_anomaly["anomaly_id"],
        "severity": "SEVERE" if live_anomaly["max_efi"] > 0.85 else "MODERATE",
        "headline": f"Live NWP Satellite Alert: {live_anomaly['name']}",
        "description": f"Real-time operational forecast detected {live_anomaly['category']} anomaly with EFI {live_anomaly['max_efi']}. Steering flow tracking active.",
        "centroid_lat": first_wp["lat"],
        "centroid_lon": first_wp["lon"],
        "radius_km": 5.0,
        "affected_districts": [live_anomaly["region"].split("(")[0].strip(), "Adjacent Coastal / Interior Sector"],
        "active": True,
        "created_at": live_anomaly["timestamp"]
    }
    db.alerts.update_one({"alert_id": alert_doc["alert_id"]}, {"$set": alert_doc}, upsert=True)
    
    return {
        "status": "success",
        "live_stream": True,
        "source": live_anomaly["source"],
        "anomaly": live_anomaly,
        "alert": alert_doc
    }

