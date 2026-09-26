"""
Physics-Informed Loss Constraints for MAUSAM.
Enforces thermodynamic conservation laws, moisture flux convergence,
geostrophic balance, and non-negativity to ensure scientific plausibility.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class AtmosphericPhysicsLoss(nn.Module):
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

    def compute_spatial_gradients(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes 2D central spatial gradients dF/dx (longitude) and dF/dy (latitude).
        field: tensor [B, C, H, W] or [H, W]
        """
        if field.dim() == 2:
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.dim() == 3:
            field = field.unsqueeze(1)
            
        # Central difference along X (W axis)
        df_dx = torch.zeros_like(field)
        df_dx[:, :, :, 1:-1] = (field[:, :, :, 2:] - field[:, :, :, :-2]) / (2.0 * self.dx)
        df_dx[:, :, :, 0] = (field[:, :, :, 1] - field[:, :, :, 0]) / self.dx
        df_dx[:, :, :, -1] = (field[:, :, :, -1] - field[:, :, :, -2]) / self.dx

        # Central difference along Y (H axis)
        df_dy = torch.zeros_like(field)
        df_dy[:, :, 1:-1, :] = (field[:, :, 2:, :] - field[:, :, :-2, :]) / (2.0 * self.dy)
        df_dy[:, :, 0, :] = (field[:, :, 1, :] - field[:, :, 0, :]) / self.dy
        df_dy[:, :, -1, :] = (field[:, :, -1, :] - field[:, :, -2, :]) / self.dy

        return df_dx, df_dy

    def moisture_continuity_loss(self, precip: torch.Tensor, u_wind: torch.Tensor, v_wind: torch.Tensor, specific_humidity: torch.Tensor) -> torch.Tensor:
        """
        Penalizes heavy downpours that lack corresponding moisture flux convergence.
        Moisture flux vector: F = (u * q, v * q)
        Convergence: - div(F) = - (d(uq)/dx + d(vq)/dy)
        """
        uq = u_wind * specific_humidity
        vq = v_wind * specific_humidity
        
        duq_dx, _ = self.compute_spatial_gradients(uq)
        _, dvq_dy = self.compute_spatial_gradients(vq)
        
        # Horizontal moisture convergence = - divergence
        moisture_convergence = -(duq_dx + dvq_dy)
        moisture_conv_positive = torch.relu(moisture_convergence)
        
        # Scale factor for physical units matching (mm/h vs kg/(m^2*s))
        scale = 3600.0 * 100.0 
        expected_precip_capacity = moisture_conv_positive * scale + 2.0 # baseline local moisture availability
        
        # Severe penalty if predicted precip exceeds physical moisture convergence capacity
        unphysical_precip_excess = torch.relu(precip - expected_precip_capacity)
        return torch.mean(unphysical_precip_excess ** 2)

    def geostrophic_balance_loss(self, u_wind: torch.Tensor, v_wind: torch.Tensor, geopotential: torch.Tensor, lats: torch.Tensor) -> torch.Tensor:
        """
        Penalizes large deviations between predicted winds and geostrophic balance:
        u_g = - (1 / f) * d(Phi)/dy
        v_g =   (1 / f) * d(Phi)/dx
        f = 2 * omega * sin(lat)
        """
        if lats.dim() == 1:
            lats_2d = lats.view(-1, 1).expand(geopotential.shape[-2], geopotential.shape[-1])
        else:
            lats_2d = lats
            
        f = 2.0 * self.omega * torch.sin(torch.deg2rad(lats_2d))
        # Mask out near-equator region (|lat| < 5 deg) where Coriolis parameter f -> 0
        coriolis_mask = (torch.abs(lats_2d) > 5.0).float()
        f_safe = torch.where(torch.abs(f) < 1e-5, torch.sign(f) * 1e-5 + 1e-6, f)

        dphi_dx, dphi_dy = self.compute_spatial_gradients(geopotential)
        
        # Geostrophic wind approximations
        u_geo = - (1.0 / f_safe) * dphi_dy
        v_geo =   (1.0 / f_safe) * dphi_dx
        
        # Compare actual winds with geostrophic expectation
        diff_u = (u_wind - u_geo) * coriolis_mask
        diff_v = (v_wind - v_geo) * coriolis_mask
        
        return torch.mean(diff_u ** 2 + diff_v ** 2) * 1e-4

    def non_negativity_loss(self, precip: torch.Tensor, humidity: torch.Tensor) -> torch.Tensor:
        """Enforces strictly non-negative values for precipitation and specific humidity."""
        precip_neg = torch.relu(-precip)
        humidity_neg = torch.relu(-humidity)
        return torch.mean(precip_neg ** 2) + torch.mean(humidity_neg ** 2)

    def forward(self, 
                precip: torch.Tensor, 
                u_wind: torch.Tensor, 
                v_wind: torch.Tensor, 
                humidity: torch.Tensor, 
                geopotential: torch.Tensor, 
                lats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculates total physics-informed loss composite.
        """
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
