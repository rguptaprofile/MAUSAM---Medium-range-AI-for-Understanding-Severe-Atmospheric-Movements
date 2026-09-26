"""
Database package for MAUSAM.
Supports MongoDB native driver with fallback document storage.
"""
from .mongo import db, get_database, init_db_indexes
from .models import (
    AnomalyTrack,
    DownscaledGrid,
    SpatialAlert,
    ForecastRun,
    DistrictSubscription
)

__all__ = [
    "db",
    "get_database",
    "init_db_indexes",
    "AnomalyTrack",
    "DownscaledGrid",
    "SpatialAlert",
    "ForecastRun",
    "DistrictSubscription"
]
