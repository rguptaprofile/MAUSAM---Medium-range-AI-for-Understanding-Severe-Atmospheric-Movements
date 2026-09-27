"""
Main FastAPI Application Entrypoint for MAUSAM.
Team Lunar - Smart India Hackathon 2026 (Problem Statement SIH26078).
"""
import logging
import os
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import settings
from .database.mongo import init_db_indexes, db
from .api import forecast_router, alerts_router, analytics_router, v1_router
from .core.pipeline import MausamPipeline
from .training import continual_learning_engine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mausam.server")

def _seed_benchmarks_background():
    """Seeds baseline benchmark scenarios and historical truth-lagged pairs in background."""
    # Never run heavy simulation pipelines in short-lived serverless environments like Vercel
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return
    try:
        if db.anomalies.count_documents({}) == 0:
            logger.info("Database empty on startup. Pre-seeding baseline benchmark scenarios (Cyclone Amphan & Heatwave)...")
            pipeline = MausamPipeline(demo_mode=True)
            pipeline.run_full_pipeline(scenario_type="cyclone_amphan")
            pipeline.run_full_pipeline(scenario_type="north_india_heatwave")
            logger.info("Baseline scenarios successfully seeded in background!")

        # Automated stream ingestion: Current live + past 3-10 days data for continuous training
        if db.training_samples.count_documents({}) < 5:
            logger.info("Ingesting operational stream (current + past 3-10 days) into continual learning store...")
            continual_learning_engine.ingest_live_and_historical_stream(days_back=10)
            logger.info("Continual learning training store populated with verified truth-lagged pairs.")
    except Exception as e:
        logger.warning(f"Background stream & scenario seeding notice: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence: Instant non-blocking execution
    logger.info("Initializing MAUSAM AI System (SIH26078 Production Core)...")
    try:
        init_db_indexes()
    except Exception as e:
        logger.warning(f"Database index initialization notice: {e}")
        
    threading.Thread(target=_seed_benchmarks_background, daemon=True).start()
    yield
    logger.info("Shutting down MAUSAM service.")

app = FastAPI(
    title="MAUSAM - Atmospheric Anomaly Tracking & Diffusion Downscaling API",
    description="SIH 2026 (SIH26078) - Team Lunar: AI-driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts (3-10 Days)",
    version="1.2.0",
    lifespan=lifespan
)

# CORS setup for web frontend and deployed Vercel domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers (SIH26078 Production v1 + Legacy Compatibility)
app.include_router(v1_router)
app.include_router(forecast_router)
app.include_router(alerts_router)
app.include_router(analytics_router)

# Mount Static Files directory for Dashboard UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_dashboard():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "project": "MAUSAM",
        "description": "Medium-range AI for Understanding Severe Atmospheric Movements",
        "team": "Team Lunar (ID: 170924)",
        "docs": "/docs",
        "status": "online"
    }

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": settings.PROJECT_NAME,
        "database": db.get_status()
    }
