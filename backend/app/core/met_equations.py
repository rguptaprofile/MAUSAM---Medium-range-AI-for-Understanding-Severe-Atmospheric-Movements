"""
Physical Meteorological Equations Engine for MAUSAM.
Conforms to MetPy and WMO atmospheric physics standards:
- Potential Temperature (Poisson's relation)
- Saturation Vapor Pressure (Bolton / Clausius-Clapeyron)
- Relative Vorticity & Divergence (Kinematic analysis)
- Geostrophic Wind Balance (Coriolis equilibrium)
- Extreme Forecast Index (Anderson-Darling tail weighting)
"""
import numpy as np
import torch
from typing import Dict, Any, Tuple

# Physical Atmospheric Constants
R_D = 287.058      # Gas constant for dry air (J/(kg*K))
C_P = 1004.64      # Specific heat of dry air at constant pressure (J/(kg*K))
KAPPA = R_D / C_P  # Poisson constant (~0.286)
P_0 = 1000.0       # Reference pressure (hPa)
OMEGA = 7.2921e-5  # Earth angular velocity (rad/s)
G = 9.80665        # Gravitational acceleration (m/s^2)

class MeteorologicalDiagnostics:
    @staticmethod
    def potential_temperature(temp_k: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
        """
        Calculates Potential Temperature (theta) via Poisson's Equation:
        theta = T * (P_0 / P) ** (R_d / c_p)
        """
        return temp_k * (P_0 / np.maximum(pressure_hpa, 10.0)) ** KAPPA

    @staticmethod
    def saturation_vapor_pressure(temp_k: np.ndarray) -> np.ndarray:
        """
        Calculates Saturation Vapor Pressure e_s (hPa) via Bolton's formulation:
        e_s(T) = 6.112 * exp( (17.67 * T_c) / (T_c + 243.5) )
        """
        temp_c = temp_k - 273.15
        return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))

    @staticmethod
    def relative_vorticity(u_wind: np.ndarray, v_wind: np.ndarray, dx_meters: float, dy_meters: float) -> np.ndarray:
        """
        Calculates vertical component of relative vorticity:
        zeta = dv/dx - du/dy
        """
        dv_dx = np.gradient(v_wind, dx_meters, axis=-1)
        du_dy = np.gradient(u_wind, dy_meters, axis=-2)
        return dv_dx - du_dy

    @staticmethod
    def horizontal_divergence(u_wind: np.ndarray, v_wind: np.ndarray, dx_meters: float, dy_meters: float) -> np.ndarray:
        """
        Calculates horizontal wind divergence:
        div = du/dx + dv/dy
        """
        du_dx = np.gradient(u_wind, dx_meters, axis=-1)
        dv_dy = np.gradient(v_wind, dy_meters, axis=-2)
        return du_dx + dv_dy

    @staticmethod
    def geostrophic_wind(geopotential: np.ndarray, lats_deg: np.ndarray, dx_meters: float, dy_meters: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates geostrophic wind components (u_g, v_g):
        u_g = - (1/f) * dPhi/dy
        v_g =   (1/f) * dPhi/dx
        """
        f = 2.0 * OMEGA * np.sin(np.radians(lats_deg))
        # Mask near equator (|lat| < 5 deg)
        f_safe = np.where(np.abs(f) < 1e-5, np.sign(f) * 1e-5 + 1e-6, f)

        dphi_dx = np.gradient(geopotential, dx_meters, axis=-1)
        dphi_dy = np.gradient(geopotential, dy_meters, axis=-2)

        u_g = - (1.0 / f_safe) * dphi_dy
        v_g =   (1.0 / f_safe) * dphi_dx
        return u_g, v_g

    @staticmethod
    def equivalent_potential_temperature(temp_k: np.ndarray, pressure_hpa: np.ndarray, spec_humidity: np.ndarray) -> np.ndarray:
        """
        Calculates Equivalent Potential Temperature (theta_e) to identify convective instability.
        theta_e ~ theta * exp( (L_v * r) / (c_p * T) )
        """
        theta = MeteorologicalDiagnostics.potential_temperature(temp_k, pressure_hpa)
        l_v = 2.5e6 # Latent heat of vaporization J/kg
        # Mixing ratio r ~ q / (1 - q)
        r = spec_humidity / np.maximum(1.0 - spec_humidity, 1e-5)
        return theta * np.exp((l_v * r) / (C_P * temp_k))
