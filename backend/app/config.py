"""
Configuration settings for MAUSAM system.
Loads settings from environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "MAUSAM: Medium-range AI for Understanding Severe Atmospheric Movements"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # MongoDB Config
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "mausam_db"
    ENABLE_MONGO_LOCAL_FALLBACK: bool = True
    
    # AI & Meteorological Parameters
    SPHERICAL_MESH_LEVEL: int = 3
    ENSEMBLE_MEMBERS: int = 21
    DIFFUSION_STEPS: int = 20
    PHYSICS_LOSS_WEIGHT: float = 0.25
    
    # Grid Specifications
    INPUT_RESOLUTION_KM: float = 12.0
    DOWNSCALED_RESOLUTION_KM: float = 5.0
    IMPACT_RADIUS_KM: float = 5.0
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
