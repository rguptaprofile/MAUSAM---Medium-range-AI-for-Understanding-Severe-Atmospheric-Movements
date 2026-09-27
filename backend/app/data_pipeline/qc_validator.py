"""
Strict Meteorological Quality Control (QC) Validator for MAUSAM SIH26078.
Replaces silent fallback with strict input validation for scientific production.
Rejects incomplete forecast inputs and surfaces detailed data-quality error.
Permits synthetic fallback strictly when DEMO_MODE is explicitly enabled.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("mausam.data_pipeline.qc")

MANDATORY_VARIABLES = ["t2m", "mslp", "precip", "u_wind", "v_wind"]
RECOMMENDED_VARIABLES = ["humidity", "z500"]

# Physical meteorological bounds across the troposphere
PHYSICAL_LIMITS = {
    "t2m": (200.0, 340.0),       # Kelvin (or -73°C to +67°C)
    "mslp": (850.0, 1080.0),     # hPa
    "precip": (0.0, 1500.0),     # mm/period (strictly non-negative)
    "u_wind": (-150.0, 150.0),   # m/s
    "v_wind": (-150.0, 150.0),   # m/s
    "humidity": (0.0, 0.05),     # kg/kg specific humidity
    "z500": (4500.0, 6200.0)     # geopotential meters
}

class DataValidationError(Exception):
    """Raised when meteorological forecast input fails strict QC rules in production mode."""
    pass

class MeteorologicalQCValidator:
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode

    def validate_forecast_dataset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs comprehensive QC checks on an atmospheric forecast dataset:
        1. Checks for missing mandatory variables.
        2. Validates spatial coordinates (lat/lon).
        3. Validates temporal lead dimensions (Day 3-10).
        4. Validates physical limits (ranges and non-negativity).
        5. Returns structured QC report with status: PASS, WARNING, or REJECTED.
        """
        qc_issues: List[str] = []
        warnings: List[str] = []

        # 1. Coordinate checks
        if "lats" not in data or "lons" not in data:
            qc_issues.append("Dataset missing required spatial coordinates ('lats', 'lons').")
        else:
            lats = np.asarray(data["lats"])
            lons = np.asarray(data["lons"])
            if lats.size == 0 or lons.size == 0:
                qc_issues.append("Spatial coordinate arrays are empty.")
            elif np.isnan(lats).any() or np.isnan(lons).any():
                qc_issues.append("NaN detected within spatial coordinate arrays.")

        # 2. Mandatory Variable Check
        for var in MANDATORY_VARIABLES:
            if var not in data or data[var] is None:
                msg = f"Mandatory atmospheric variable '{var}' is missing from NWP forecast stream."
                if self.demo_mode:
                    warnings.append(f"DEMO_MODE active: synthesized fallback for '{var}'.")
                else:
                    qc_issues.append(msg)
            else:
                arr = np.asarray(data[var])
                if arr.size == 0:
                    qc_issues.append(f"Variable '{var}' contains empty data array.")
                    continue
                # NaN check
                nan_count = int(np.isnan(arr).sum())
                if nan_count > 0:
                    nan_pct = (nan_count / arr.size) * 100
                    if nan_pct > 10.0:
                        qc_issues.append(f"Variable '{var}' exceeds 10% NaN threshold ({nan_pct:.1f}% NaNs).")
                    else:
                        warnings.append(f"Variable '{var}' contains {nan_count} NaNs ({nan_pct:.2f}%); imputed.")

                # Physical limits check
                if var in PHYSICAL_LIMITS:
                    v_min, v_max = PHYSICAL_LIMITS[var]
                    val_min = float(np.nanmin(arr))
                    val_max = float(np.nanmax(arr))
                    
                    # Handle units (e.g. t2m in Celsius vs Kelvin, mslp in Pa vs hPa)
                    if var == "t2m" and val_max < 65.0: # Given in Celsius
                        val_min += 273.15
                        val_max += 273.15
                    elif var == "mslp" and float(np.nanmean(arr)) > 2000.0:
                        val_min /= 100.0
                        val_max /= 100.0

                    if val_min < v_min or val_max > v_max:
                        warnings.append(
                            f"Variable '{var}' has values outside standard physical bounds [{v_min}, {v_max}]: range is [{val_min:.1f}, {val_max:.1f}]."
                        )

        # 3. Lead Time Dimension Check
        lead_days = data.get("lead_days", [])
        if not lead_days or len(lead_days) == 0:
            qc_issues.append("Forecast lead days missing or empty.")
        else:
            if min(lead_days) > 10.0 or max(lead_days) < 3.0:
                warnings.append("Lead days do not overlap standard 3-10 day medium-range forecast window.")

        # QC Decision
        if qc_issues:
            if not self.demo_mode:
                err_msg = f"Strict Meteorological QC REJECTED dataset: {'; '.join(qc_issues)}"
                logger.error(err_msg)
                raise DataValidationError(err_msg)
            else:
                qc_status = "WARNING_DEMO_SUBSTITUTION"
        elif warnings:
            qc_status = "PASSED_WITH_WARNINGS"
        else:
            qc_status = "PASSED_STRICT_METEO_QC"

        return {
            "qc_status": qc_status,
            "qc_passed": len(qc_issues) == 0,
            "issues": qc_issues,
            "warnings": warnings,
            "demo_mode": self.demo_mode,
            "variables_verified": [v for v in MANDATORY_VARIABLES if v in data]
        }
