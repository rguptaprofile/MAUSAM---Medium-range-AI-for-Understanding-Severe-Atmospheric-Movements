"""
Spatial Alerting and NDRF Dispatch API for MAUSAM.
Outputs pinpoint coordinates, 5km impact radii, categorized spatial alerts (Low, Moderate, Severe),
and GeoJSON layers for GIS / First Responder integration.
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
import numpy as np
import uuid
from datetime import datetime
from ..database.mongo import db
from ..database.models import SpatialAlert, DistrictSubscription

router = APIRouter(prefix="/api/alerts", tags=["Spatial Alerts & NDRF Dispatch"])

def generate_circle_geojson(lat: float, lon: float, radius_km: float = 5.0, num_points: int = 36) -> Dict[str, Any]:
    """Generates a GeoJSON polygon approximating a 5km radius impact circle."""
    # Approximate 1 deg lat ~ 111.0 km, 1 deg lon ~ 111.0 * cos(lat) km
    r_lat = radius_km / 111.0
    r_lon = radius_km / (111.0 * np.cos(np.radians(lat)) + 1e-6)
    
    angles = np.linspace(0, 2 * np.pi, num_points)
    coordinates = []
    for a in angles:
        p_lat = lat + r_lat * np.sin(a)
        p_lon = lon + r_lon * np.cos(a)
        coordinates.append([round(float(p_lon), 6), round(float(p_lat), 6)])
    coordinates.append(coordinates[0]) # close polygon
    
    return {
        "type": "Polygon",
        "coordinates": [coordinates]
    }

@router.get("/active")
def get_active_alerts(severity: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns active spatial alerts with exact centroid coordinates,
    5 km radius impact zones, and actionable threat levels.
    """
    query = {"active": True}
    if severity:
        query["severity"] = severity.upper()
    return db.alerts.find(query, sort=[("created_at", -1)])

@router.get("/geojson")
def get_alerts_geojson() -> Dict[str, Any]:
    """
    Exports all active spatial alerts as a standardized GeoJSON FeatureCollection.
    Features include:
    1. Pinpoint centroid Point feature.
    2. Exact 5 km radius impact Polygon feature.
    Compatible with Leaflet, Mapbox, QGIS, and ArcGIS.
    """
    active_alerts = db.alerts.find({"active": True})
    features = []
    
    for alert in active_alerts:
        lat = alert["centroid_lat"]
        lon = alert["centroid_lon"]
        radius = alert.get("impact_radius_km", 5.0)
        
        # 1. Point Feature (Pinpoint coordinate)
        point_feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "alert_id": alert["alert_id"],
                "type": "centroid",
                "severity": alert["severity"],
                "headline": alert["headline"],
                "lead_time": alert["lead_time_window"],
                "affected_districts": alert.get("affected_districts", [])
            }
        }
        features.append(point_feat)
        
        # 2. Polygon Feature (5 km circular impact zone)
        polygon_geom = generate_circle_geojson(lat, lon, radius_km=radius)
        poly_feat = {
            "type": "Feature",
            "geometry": polygon_geom,
            "properties": {
                "alert_id": alert["alert_id"],
                "type": "5km_impact_radius",
                "severity": alert["severity"],
                "radius_km": radius,
                "ndrf_recommended": alert.get("ndrf_deployment_recommended", False)
            }
        }
        features.append(poly_feat)
        
    return {
        "type": "FeatureCollection",
        "metadata": {
            "system": "MAUSAM AI Early Warning Core",
            "generated_at": datetime.utcnow().isoformat(),
            "count": len(features)
        },
        "features": features
    }

@router.post("/dispatch-ndrf/{alert_id}")
def trigger_ndrf_dispatch(alert_id: str, notes: Optional[str] = Body(default="Automated tactical deployment triggered via MAUSAM API")) -> Dict[str, Any]:
    """
    Simulates programmatic dispatch of NDRF / First Responder task forces
    to the pinpoint 5 km anomaly centroid coordinates.
    """
    alert = db.alerts.find_one({"alert_id": alert_id})
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
    dispatch_record = {
        "dispatch_id": f"NDRF-DISPATCH-{uuid.uuid4().hex[:8].upper()}",
        "alert_id": alert_id,
        "target_coordinates": {
            "latitude": alert["centroid_lat"],
            "longitude": alert["centroid_lon"]
        },
        "severity": alert["severity"],
        "assigned_battalion": "10th Bn NDRF (Eastern Command)" if alert["centroid_lon"] > 82.0 else "8th Bn NDRF (Northern/Western)",
        "tactical_orders": [
            f"Pre-deploy flood rescue inflatable motorized boats within 5 km impact radius of [{alert['centroid_lat']}N, {alert['centroid_lon']}E]",
            "Coordinate with District Disaster Management Authority (DDMA)",
            "Initiate selective evacuation of identified subgrid hamlets"
        ],
        "status": "DISPATCH_CONFIRMED",
        "dispatched_at": datetime.utcnow().isoformat()
    }
    
    # Log in audit
    db.audit_logs.insert_one(dispatch_record)
    
    return {
        "status": "success",
        "message": f"NDRF tactical deployment successfully confirmed for alert {alert_id}",
        "dispatch_details": dispatch_record
    }

@router.post("/subscribe")
def register_subscription(sub: DistrictSubscription) -> Dict[str, Any]:
    """Registers an emergency responder or agricultural authority for hyper-local 5 km alerts."""
    doc = sub.model_dump()
    db.subscriptions.insert_one(doc)
    return {
        "status": "success",
        "message": f"Subscription registered for {sub.district_name} ({sub.authority})",
        "subscription_id": sub.id
    }

@router.get("/subscriptions")
def list_subscriptions() -> List[Dict[str, Any]]:
    """Lists registered disaster authorities and farmer collectives."""
    return db.subscriptions.find({})
