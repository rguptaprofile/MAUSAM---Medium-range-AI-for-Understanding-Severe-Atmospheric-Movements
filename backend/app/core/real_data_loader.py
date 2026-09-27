"""
Real Meteorological Data Loader for MAUSAM.
Processes actual NetCDF (.nc) and GRIB2 files from:
- NCMRWF NEPS-G (12 km Global Ensemble)
- ECMWF Open Data / ERA5 Reanalysis
- IMDAA Regional Reanalysis
"""
try:
    import xarray as xr
except ImportError:
    xr = None
import numpy as np
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("mausam.real_data")

class RealAtmosphericDataLoader:
    def __init__(self):
        pass

    def load_netcdf(self, file_path: str, demo_mode: bool = False) -> Dict[str, Any]:
        """
        Reads a standard meteorological NetCDF file using Xarray and normalizes variables
        into the format expected by MAUSAM's Spherical GNN and Diffusion pipeline.
        In production mode (demo_mode=False), rejects incomplete inputs instead of silent synthesis.
        """
        if xr is None:
            raise ImportError("xarray and netCDF4 are required for reading NetCDF datasets. Install via: pip install xarray netCDF4")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NetCDF file not found at: {file_path}")

        logger.info(f"Opening real atmospheric NetCDF dataset: {file_path} (demo_mode={demo_mode})")
        ds = xr.open_dataset(file_path)

        # 1. Resolve Coordinate Names
        lat_coord = None
        for name in ["latitude", "lat", "LAT", "lats"]:
            if name in ds.coords or name in ds.dims:
                lat_coord = name
                break

        lon_coord = None
        for name in ["longitude", "lon", "LON", "lons"]:
            if name in ds.coords or name in ds.dims:
                lon_coord = name
                break

        if lat_coord is None or lon_coord is None:
            raise ValueError("Dataset missing standard latitude/longitude coordinates")

        lats = ds[lat_coord].values
        lons = ds[lon_coord].values

        # If 1D coordinates, create 2D meshgrid
        if lats.ndim == 1 and lons.ndim == 1:
            LATS, LONS = np.meshgrid(lats, lons, indexing="ij")
        else:
            LATS, LONS = lats, lons

        # Check for ensemble member dimension in metadata
        member_coord = None
        for m_name in ["member", "number", "ens", "realization"]:
            if m_name in ds.coords or m_name in ds.dims:
                member_coord = m_name
                break
        num_members = len(ds[member_coord]) if member_coord else 1

        # 2. Extract or Map Atmospheric Variables with standard fallbacks
        var_map = {
            "u_wind": ["u10", "u", "10u", "u_wind", "UGRD"],
            "v_wind": ["v10", "v", "10v", "v_wind", "VGRD"],
            "t2m": ["t2m", "t", "2t", "temperature", "TMP"],
            "mslp": ["msl", "mslp", "pres", "pressure", "PRMSL"],
            "precip": ["tp", "precip", "total_precipitation", "APCP", "pr"],
            "humidity": ["q", "r", "humidity", "specific_humidity", "SPFH"],
            "z500": ["z", "gh", "geopotential", "hgt", "HGT"]
        }

        mandatory_in_prod = ["u_wind", "v_wind", "t2m", "mslp", "precip"]
        extracted = {}
        for target_var, aliases in var_map.items():
            matched_var = None
            for alias in aliases:
                if alias in ds.data_vars:
                    matched_var = alias
                    break
            
            if matched_var:
                arr = ds[matched_var].values
                # Standardize units:
                if target_var == "mslp" and np.nanmean(arr) > 2000.0:
                    arr = arr / 100.0 # Convert Pa to hPa
                elif target_var == "precip" and np.nanmax(arr) < 1.0:
                    arr = arr * 1000.0 # Convert m to mm
                extracted[target_var] = np.nan_to_num(arr, nan=0.0).astype(np.float32)
            else:
                if not demo_mode and target_var in mandatory_in_prod:
                    ds.close()
                    raise ValueError(f"Production NetCDF ingest rejected: missing mandatory atmospheric variable '{target_var}'. Set demo_mode=True to permit synthetic substitution.")
                logger.warning(f"Variable '{target_var}' not found in NetCDF; substituting neutral physical baseline (DEMO_MODE={demo_mode}).")
                num_steps = 8
                shape = (num_steps, LATS.shape[0], LATS.shape[1])
                if target_var == "precip":
                    extracted[target_var] = np.zeros(shape, dtype=np.float32)
                elif target_var == "t2m":
                    extracted[target_var] = np.full(shape, 300.0, dtype=np.float32)
                elif target_var == "mslp":
                    extracted[target_var] = np.full(shape, 1010.0, dtype=np.float32)
                elif target_var in ["u_wind", "v_wind"]:
                    extracted[target_var] = np.zeros(shape, dtype=np.float32)
                elif target_var == "humidity":
                    extracted[target_var] = np.full(shape, 0.012, dtype=np.float32)
                elif target_var == "z500":
                    extracted[target_var] = np.full(shape, 5800.0, dtype=np.float32)

        # 3. Handle Time Steps
        time_coord = None
        for name in ["time", "valid_time", "step", "forecast_reference_time"]:
            if name in ds.coords or name in ds.dims:
                time_coord = name
                break

        lead_days = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        if time_coord and len(ds[time_coord]) >= 8:
            lead_days = [float(3.0 + i) for i in range(8)]

        ds.close()

        return {
            "source_file": os.path.basename(file_path),
            "ensemble_members": num_members,
            "lead_days": lead_days,
            "lats": LATS,
            "lons": LONS,
            "u_wind": extracted["u_wind"],
            "v_wind": extracted["v_wind"],
            "t2m": extracted["t2m"],
            "mslp": extracted["mslp"],
            "precip": extracted["precip"],
            "humidity": extracted["humidity"],
            "z500": extracted["z500"]
        }

    def generate_sample_real_netcdf(self, output_path: str):
        """
        Creates a valid sample NetCDF-4 file conforming to CF-1.8 conventions
        for testing and verification with real NetCDF readers.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lats = np.linspace(10.0, 30.0, 60, dtype=np.float32)
        lons = np.linspace(70.0, 90.0, 60, dtype=np.float32)
        times = [datetime(2026, 5, 20) + timedelta(days=float(d)) for d in [3, 4, 5, 6, 7, 8, 9, 10]]

        # Synthetic cyclonic depression
        LATS, LONS = np.meshgrid(lats, lons, indexing="ij")
        tp_data = np.zeros((8, 60, 60), dtype=np.float32)
        u_data = np.zeros((8, 60, 60), dtype=np.float32)
        v_data = np.zeros((8, 60, 60), dtype=np.float32)
        t2m_data = np.full((8, 60, 60), 302.0, dtype=np.float32)
        mslp_data = np.full((8, 60, 60), 1008.0, dtype=np.float32)
        q_data = np.full((8, 60, 60), 0.012, dtype=np.float32)
        z_data = np.full((8, 60, 60), 5820.0, dtype=np.float32)

        for i, t in enumerate(times):
            c_lat = 14.0 + i * 1.5
            c_lon = 84.0 + i * 0.7
            dist_sq = (LATS - c_lat)**2 + (LONS - c_lon)**2
            vortex = np.exp(-dist_sq / 8.0)
            
            # Extreme amplitudes matching severe cyclone
            tp_data[i] = 1.0 + vortex * 85.0 # up to 86 mm/h
            u_data[i] = - (LATS - c_lat) * 25.0 * vortex
            v_data[i] = (LONS - c_lon) * 25.0 * vortex
            mslp_data[i] -= vortex * 75.0 # drops to ~933 hPa
            q_data[i] += vortex * 0.010
            z_data[i] -= vortex * 220.0

        ds = xr.Dataset(
            data_vars={
                "tp": (["time", "latitude", "longitude"], tp_data, {"units": "mm", "long_name": "Total Precipitation"}),
                "u10": (["time", "latitude", "longitude"], u_data, {"units": "m s-1", "long_name": "10m U-Wind Component"}),
                "v10": (["time", "latitude", "longitude"], v_data, {"units": "m s-1", "long_name": "10m V-Wind Component"}),
                "t2m": (["time", "latitude", "longitude"], t2m_data, {"units": "K", "long_name": "2m Temperature"}),
                "msl": (["time", "latitude", "longitude"], mslp_data * 100.0, {"units": "Pa", "long_name": "Mean Sea Level Pressure"}),
                "q": (["time", "latitude", "longitude"], q_data, {"units": "kg kg-1", "long_name": "Specific Humidity"}),
                "z": (["time", "latitude", "longitude"], z_data, {"units": "m2 s-2", "long_name": "Geopotential Height"})
            },
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons
            },
            attrs={
                "title": "NCMRWF NEPS-G 12km Global Ensemble Anomaly Slice (Sample)",
                "institution": "NCMRWF / IMD Open Data Stream",
                "conventions": "CF-1.8",
                "project": "MAUSAM SIH 2026"
            }
        )

        ds.to_netcdf(output_path, format="NETCDF4")
        logger.info(f"Sample real NetCDF file created at: {output_path}")
        return output_path
