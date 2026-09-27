"""
MAUSAM Data Sources Package (SIH26078).
Exposes real NCMRWF NEPS-G, NCUM-G, ERA5 Climatology, IMDAA, IMD, ECMWF, and HighRes connectors.
"""
from .base_connector import BaseDataSourceConnector
from .neps_g_connector import NEPSGConnector
from .ncum_g_connector import NCUMGConnector
from .era5_baseline_connector import ERA5BaselineConnector
from .imdaa_connector import IMDAAConnector
from .imd_api_connector import IMDAPIConnector
from .ecmwf_open_connector import ECMWFOpenConnector
from .highres_regional_connector import HighResRegionalConnector

# Global singleton connectors
neps_g_source = NEPSGConnector()
ncum_g_source = NCUMGConnector()
era5_baseline_source = ERA5BaselineConnector()
imdaa_source = IMDAAConnector()
imd_api_source = IMDAPIConnector()
ecmwf_open_source = ECMWFOpenConnector()
highres_source = HighResRegionalConnector()

ALL_CONNECTORS = {
    "neps_g": neps_g_source,
    "ncum_g": ncum_g_source,
    "era5_baseline": era5_baseline_source,
    "imdaa": imdaa_source,
    "imd_api": imd_api_source,
    "ecmwf_benchmark": ecmwf_open_source,
    "highres_regional": highres_source
}

def get_all_sources_status():
    """Polls health status and metadata across all configured data sources."""
    status_report = []
    for key, conn in ALL_CONNECTORS.items():
        status_report.append(conn.health_check())
    return status_report
