"""
MongoDB Setup, Indexing, and Benchmark Seeding Script for MAUSAM.
Team Lunar - Smart India Hackathon 2026 (SIH26078).
"""
import sys
import os
from datetime import datetime

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.config import settings
from backend.app.database.mongo import db, init_db_indexes
from backend.app.core.pipeline import MausamPipeline
from backend.app.database.models import DistrictSubscription

def setup_database():
    print("=" * 65)
    print("   PROJECT MAUSAM - MONGODB INITIALIZATION & BENCHMARK SEEDER")
    print("=" * 65)
    print(f"Target URI: {settings.MONGODB_URI}")
    print(f"Database:   {settings.MONGODB_DB_NAME}")
    print(f"Mode:       {db.get_status()['mode']}")
    print("-" * 65)

    # 1. Initialize Indexes
    print("[1/4] Creating MongoDB compound and spatial 2dsphere indexes...")
    init_db_indexes()
    print("  -> Indexes successfully applied.")

    # 2. Register Subscriptions (NDRF, State Disaster Authorities, Farmer KVKs)
    print("[2/4] Registering emergency responder and agricultural subscriptions...")
    sample_subscriptions = [
        DistrictSubscription(
            district_name="South 24 Parganas",
            state="West Bengal",
            authority="NDRF 2nd Battalion (Haringhata) / WB SDRF",
            contact_email="ndrf.wb@nic.in",
            phone="+91-9433001122",
            lat=22.1667,
            lon=88.5833,
            alert_threshold="SEVERE"
        ),
        DistrictSubscription(
            district_name="Kendrapara",
            state="Odisha",
            authority="Odisha State Disaster Management Authority (OSDMA)",
            contact_email="osdma.kendrapara@odisha.gov.in",
            phone="+91-9437002233",
            lat=20.5000,
            lon=86.4200,
            alert_threshold="MODERATE"
        ),
        DistrictSubscription(
            district_name="Churu",
            state="Rajasthan",
            authority="Krishi Vigyan Kendra (KVK) - Agro Advisory Cell",
            contact_email="kvk.churu@icar.gov.in",
            phone="+91-9414003344",
            lat=28.2900,
            lon=74.9600,
            alert_threshold="SEVERE"
        ),
        DistrictSubscription(
            district_name="Mumbai Suburban",
            state="Maharashtra",
            authority="Municipal Corporation of Greater Mumbai (MCGM Disaster Cell)",
            contact_email="disaster@mcgm.gov.in",
            phone="+91-9820004455",
            lat=19.0760,
            lon=72.8777,
            alert_threshold="SEVERE"
        )
    ]

    for sub in sample_subscriptions:
        db.subscriptions.update_one(
            {"district_name": sub.district_name},
            {"$set": sub.model_dump()},
            upsert=True
        )
    print(f"  -> {len(sample_subscriptions)} Disaster Authority & Agro Subscriptions stored.")

    # 3. Seed Benchmark Scenarios
    print("[3/4] Running end-to-end MAUSAM pipeline to seed benchmark scenarios...")
    pipeline = MausamPipeline()

    scenarios = ["cyclone_amphan", "north_india_heatwave", "monsoon_cloudburst", "north_india_coldwave"]
    for scn in scenarios:
        print(f"  -> Ingesting, tracking, and downscaling scenario: {scn}...")
        res = pipeline.run_full_pipeline(scenario_type=scn)
        print(f"     [OK] Run ID: {res['run_id']} | Anomalies: {len(res['anomalies'])} | Alerts: {len(res['alerts'])}")

    # 4. Verification & Status Report
    print("[4/4] Verifying database integrity and collection counts...")
    status = db.get_status()
    print("-" * 65)
    print("MONGODB STATUS & TELEMETRY:")
    print(f"  Storage Provider: {status['mode']}")
    print(f"  Forecast Runs:    {status['counts']['forecast_runs']}")
    print(f"  Anomalies:        {status['counts']['anomalies']}")
    print(f"  Downscaled Grids: {status['counts']['downscaled_grids']}")
    print(f"  Spatial Alerts:   {status['counts']['alerts']}")
    print(f"  Subscriptions:    {status['counts']['subscriptions']}")
    print("=" * 65)
    print("MongoDB setup and benchmark seeding complete!")

if __name__ == "__main__":
    setup_database()
