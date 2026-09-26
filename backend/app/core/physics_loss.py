"""
Physics-Informed Loss Constraints for MAUSAM.
Enforces thermodynamic conservation laws, moisture flux convergence,
geostrophic balance, and non-negativity to ensure scientific plausibility.
Supports both PyTorch GPU/CPU and lightweight NumPy/SciPy environments.
"""
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    ModuleBase = nn.Module
except ImportError:
    HAS_TORCH = False
    class ModuleBase:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

import numpy as np
from typing import Dict, Tuple, Any

class AtmosphericPhysicsLoss(ModuleBase):
    def __init__(self, omega: float = 7.2921e-5, g: float = 9.80665, grid_spacing_km: float = 5.0):
        """
        omega: Earth rotation rate (rad/s)
        g: standard gravity (m/s^2)
        grid_spacing_km: spatial resolution between grid points in km
        """
        super().__init__()
        self.omega = omega
        self.g = g
        self.dx = grid_spacing_km * 1000.0 # convert to meters
        self.dy = grid_spacing_km * 1000.0

    def compute_spatial_gradients(self, field: Any) -> Tuple[Any, Any]:
        """
        Computes 2D central spatial gradients dF/dx (longitude) and dF/dy (latitude).
        """
        if HAS_TORCH and isinstance(field, torch.Tensor):
            if field.dim() == 2:
                field = field.unsqueeze(0).unsqueeze(0)
            elif field.dim() == 3:
                field = field.unsqueeze(1)
                
            df_dx = torch.zeros_like(field)
            df_dx[:, :, :, 1:-1] = (field[:, :, :, 2:] - field[:, :, :, :-2]) / (2.0 * self.dx)
            df_dx[:, :, :, 0] = (field[:, :, :, 1] - field[:, :, :, 0]) / self.dx
            df_dx[:, :, :, -1] = (field[:, :, :, -1] - field[:, :, :, -2]) / self.dx

            df_dy = torch.zeros_like(field)
            df_dy[:, :, 1:-1, :] = (field[:, :, 2:, :] - field[:, :, :-2, :]) / (2.0 * self.dy)
            df_dy[:, :, 0, :] = (field[:, :, 1, :] - field[:, :, 0, :]) / self.dy
            df_dy[:, :, -1, :] = (field[:, :, -1, :] - field[:, :, -2, :]) / self.dy
            return df_dx, df_dy
        else:
            arr = np.asarray(field, dtype=np.float32)
            if arr.ndim > 2:
                arr = arr.squeeze()
            gy, gx = np.gradient(arr, self.dy, self.dx)
            return gx, gy

    def moisture_continuity_loss(self, precip: Any, u_wind: Any, v_wind: Any, specific_humidity: Any) -> Any:
        """
        Penalizes heavy downpours that lack corresponding moisture flux convergence.
        """
        if HAS_TORCH and isinstance(precip, torch.Tensor):
            uq = u_wind * specific_humidity
            vq = v_wind * specific_humidity
            duq_dx, _ = self.compute_spatial_gradients(uq)
            _, dvq_dy = self.compute_spatial_gradients(vq)
            moist_convergence = - (duq_dx + dvq_dy)
            precip_rate_kg_m2_s = precip / 3600.0
            error = precip_rate_kg_m2_s - moist_convergence
            return torch.mean(error ** 2) * 1e6
        else:
            uq = np.asarray(u_wind) * np.asarray(specific_humidity)
            vq = np.asarray(v_wind) * np.asarray(specific_humidity)
            duq_dx, _ = self.compute_spatial_gradients(uq)
            _, dvq_dy = self.compute_spatial_gradients(vq)
            moist_conv = - (duq_dx + dvq_dy)
            error = (np.asarray(precip) / 3600.0) - moist_conv
            return float(np.mean(error ** 2) * 1e6)

    def geostrophic_balance_loss(self, u_wind: Any, v_wind: Any, geopotential: Any, lats: Any) -> Any:
        """
        Calculates deviation between predicted winds and geostrophic equilibrium.
        """
        if HAS_TORCH and isinstance(u_wind, torch.Tensor):
            deg2rad = torch.pi / 180.0
            f = 2.0 * self.omega * torch.sin(lats * deg2rad)
            if f.dim() == 1:
                f = f.view(-1, 1).expand_as(u_wind)
                lats_2d = lats.view(-1, 1).expand_as(u_wind)
            else:
                lats_2d = lats

            coriolis_mask = (torch.abs(lats_2d) > 5.0).float()
            f_safe = torch.where(torch.abs(f) < 1e-5, torch.sign(f) * 1e-5 + 1e-6, f)

            dphi_dx, dphi_dy = self.compute_spatial_gradients(geopotential)
            u_geo = - (1.0 / f_safe) * dphi_dy
            v_geo =   (1.0 / f_safe) * dphi_dx
            
            diff_u = (u_wind - u_geo) * coriolis_mask
            diff_v = (v_wind - v_geo) * coriolis_mask
            return torch.mean(diff_u ** 2 + diff_v ** 2) * 1e-4
        else:
            lats_arr = np.asarray(lats, dtype=np.float32)
            f = 2.0 * self.omega * np.sin(np.radians(lats_arr))
            f_safe = np.where(np.abs(f) < 1e-5, np.sign(f) * 1e-5 + 1e-6, f)
            dphi_dx, dphi_dy = self.compute_spatial_gradients(geopotential)
            u_geo = - (1.0 / f_safe) * dphi_dy
            v_geo = (1.0 / f_safe) * dphi_dx
            mask = (np.abs(lats_arr) > 5.0).astype(np.float32)
            return float(np.mean(((np.asarray(u_wind) - u_geo) * mask)**2 + ((np.asarray(v_wind) - v_geo) * mask)**2) * 1e-4)

    def non_negativity_loss(self, precip: Any, humidity: Any) -> Any:
        """Enforces strictly non-negative values for precipitation and specific humidity."""
        if HAS_TORCH and isinstance(precip, torch.Tensor):
            precip_neg = torch.relu(-precip)
            humidity_neg = torch.relu(-humidity)
            return torch.mean(precip_neg ** 2) + torch.mean(humidity_neg ** 2)
        else:
            p_neg = np.maximum(0, -np.asarray(precip))
            q_neg = np.maximum(0, -np.asarray(humidity))
            return float(np.mean(p_neg**2) + np.mean(q_neg**2))

    def forward(self, 
                precip: Any, 
                u_wind: Any, 
                v_wind: Any, 
                humidity: Any, 
                geopotential: Any, 
                lats: Any) -> Dict[str, Any]:
        """Calculates total physics-informed loss composite."""
        l_moist = self.moisture_continuity_loss(precip, u_wind, v_wind, humidity)
        l_geo = self.geostrophic_balance_loss(u_wind, v_wind, geopotential, lats)
        l_nonneg = self.non_negativity_loss(precip, humidity)
        
        total_loss = l_moist + 0.1 * l_geo + 10.0 * l_nonneg
        return {
            "total_physics_loss": total_loss,
            "moisture_continuity_loss": l_moist,
            "geostrophic_loss": l_geo,
            "non_negativity_loss": l_nonneg
        }
