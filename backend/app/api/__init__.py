"""
API routes package for MAUSAM backend.
"""
from .routes_forecast import router as forecast_router
from .routes_alerts import router as alerts_router
from .routes_analytics import router as analytics_router

__all__ = ["forecast_router", "alerts_router", "analytics_router"]
