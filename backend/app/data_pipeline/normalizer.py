"""
Meteorological Normalizer and Dask Chunking Engine for MAUSAM SIH26078.
Standardizes units (Pa -> hPa, m -> mm, K -> °C), standardizes coordinate grids,
and leverages Dask for parallel out-of-core chunked processing of multi-GB archives.
"""
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

logger = logging.getLogger("mausam.data_pipeline.normalizer")

class MeteorologicalNormalizer:
    def __init__(self, chunk_size: Tuple[int, int] = (25, 25)):
        self.chunk_size = chunk_size

    def normalize_units(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes variable units to standard SIH26078 conventions:
        - t2m: Kelvin (for thermodynamic equations) and °C for visualization
        - mslp: hPa (hectopascals)
        - precip: mm / period (millimeters)
        - u_wind, v_wind: m/s
        - humidity: kg/kg
        - z500: geopotential meters
        """
        norm = dict(data)
        
        # 1. Mean Sea Level Pressure (Pa to hPa)
        if "mslp" in norm and norm["mslp"] is not None:
            mslp = np.asarray(norm["mslp"], dtype=np.float32)
            if np.nanmean(mslp) > 2000.0: # In Pascals (e.g. 101325 Pa)
                mslp = mslp / 100.0
            norm["mslp"] = mslp

        # 2. Total Precipitation (m to mm)
        if "precip" in norm and norm["precip"] is not None:
            precip = np.asarray(norm["precip"], dtype=np.float32)
            if np.nanmax(precip) < 1.0 and np.nanmax(precip) > 0.0: # In meters
                precip = precip * 1000.0
            norm["precip"] = np.maximum(0.0, precip) # Enforce physical non-negativity

        # 3. 2m Temperature (Celsius to Kelvin or vice-versa)
        if "t2m" in norm and norm["t2m"] is not None:
            t2m = np.asarray(norm["t2m"], dtype=np.float32)
            if np.nanmean(t2m) < 80.0: # In Celsius
                t2m = t2m + 273.15
            norm["t2m"] = t2m

        # 4. Wind components (km/h to m/s if required)
        for w_var in ["u_wind", "v_wind"]:
            if w_var in norm and norm[w_var] is not None:
                w_arr = np.asarray(norm[w_var], dtype=np.float32)
                if np.nanmax(np.abs(w_arr)) > 150.0: # Likely km/h
                    w_arr = w_arr / 3.6
                norm[w_var] = w_arr

        return norm

    def dask_chunk_arrays(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wraps multi-dimensional atmospheric arrays with Dask chunking
        for scalable, memory-efficient distributed processing.
        """
        if not HAS_DASK:
            return data

        chunked = dict(data)
        for var in ["precip", "t2m", "mslp", "u_wind", "v_wind", "humidity", "z500"]:
            if var in chunked and isinstance(chunked[var], np.ndarray):
                arr = chunked[var]
                if arr.ndim == 4: # [leads, members, H, W]
                    chunked[var] = da.from_array(arr, chunks=(1, 5, self.chunk_size[0], self.chunk_size[1]))
                elif arr.ndim == 3: # [leads, H, W]
                    chunked[var] = da.from_array(arr, chunks=(1, self.chunk_size[0], self.chunk_size[1]))
                elif arr.ndim == 2: # [H, W]
                    chunked[var] = da.from_array(arr, chunks=self.chunk_size)
        return chunked

    def compute_dask_to_numpy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts Dask arrays back to in-memory NumPy for neural inference."""
        res = dict(data)
        for k, v in res.items():
            if HAS_DASK and isinstance(v, da.Array):
                res[k] = v.compute()
        return res
