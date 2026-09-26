"""
Stage 1: Spherical Graph Neural Network (GNN) Anomaly Tracker.
Executes message passing on an icosahedral mesh to isolate extreme weather anomalies
via the Extreme Forecast Index (EFI) and computes dynamic 4D spatio-temporal bounding boxes.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from .spherical_mesh import IcosahedralMesh
from datetime import datetime, timedelta

def compute_extreme_forecast_index(ensemble_values: np.ndarray, climatology_percentiles: np.ndarray) -> float:
    """
    Computes ECMWF Extreme Forecast Index (EFI).
    ensemble_values: [num_members] forecast values
    climatology_percentiles: [99] percentiles of the 30-year historical ERA5 baseline (p=0.01 to 0.99)
    Formula:
        EFI = (2 / pi) * integral_0^1 [ (p - F_f(x_p)) / sqrt(p*(1-p)) ] dp
    """
    p_values = np.linspace(0.01, 0.99, 99)
    weights = 1.0 / np.sqrt(p_values * (1.0 - p_values) + 1e-6)
    
    # Cumulative distribution function of forecast ensemble at each climatological percentile threshold
    f_forecast = np.zeros_like(p_values)
    for i, thresh in enumerate(climatology_percentiles):
        # Fraction of ensemble members <= threshold
        f_forecast[i] = np.mean(ensemble_values <= thresh)
        
    diff = p_values - f_forecast
    integrand = diff * weights
    efi = (2.0 / np.pi) * np.trapezoid(integrand, p_values)
    return float(np.clip(efi, -1.0, 1.0))

class SphericalMeshConv(nn.Module):
    """Message passing layer on icosahedral geodesic edges."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU()
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        """
        src = edge_index[0]
        dst = edge_index[1]
        
        # Edge message
        edge_feats = torch.cat([x[src], x[dst]], dim=-1)
        messages = self.message_mlp(edge_feats)
        
        # Aggregate messages at destination nodes
        num_nodes = x.size(0)
        agg_messages = torch.zeros(num_nodes, messages.size(-1), device=x.device)
        agg_messages.index_add_(0, dst, messages)
        
        # Update node representation
        out = self.update_mlp(torch.cat([x, agg_messages], dim=-1))
        return out

class SphericalGNNAnomalyTracker(nn.Module):
    def __init__(self, in_features: int = 7, hidden_dim: int = 64, mesh_level: int = 3):
        """
        in_features: [u_wind, v_wind, t2m, mslp, precip, humidity, z500]
        """
        super().__init__()
        self.mesh = IcosahedralMesh(subdivision_level=mesh_level)
        self.edge_index = self.mesh.edge_index
        
        # Multi-layer Spherical Message Passing
        self.encoder = nn.Linear(in_features, hidden_dim)
        self.conv1 = SphericalMeshConv(hidden_dim, hidden_dim)
        self.conv2 = SphericalMeshConv(hidden_dim, hidden_dim)
        self.conv3 = SphericalMeshConv(hidden_dim, hidden_dim)
        
        # Prediction heads: anomaly probability and EFI anomaly magnitude
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.efi_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        node_features: [num_nodes, in_features]
        Returns:
            anomaly_prob: [num_nodes, 1] probability of severe anomaly
            efi_pred: [num_nodes, 1] predicted Extreme Forecast Index
        """
        h = torch.relu(self.encoder(node_features))
        h = h + self.conv1(h, self.edge_index)
        h = h + self.conv2(h, self.edge_index)
        h = h + self.conv3(h, self.edge_index)
        
        anomaly_prob = self.anomaly_head(h)
        efi_pred = self.efi_regressor(h)
        return anomaly_prob, efi_pred

    def track_forecast_trajectory(self, 
                                  ensemble_forecast_4d: Dict[str, np.ndarray], 
                                  lead_days: List[float]) -> List[Dict[str, Any]]:
        """
        Tracks anomalies over a 3 to 10-day forecast horizon.
        Identifies high EFI centroids, groups them across consecutive lead days,
        and computes macroscale 4D spatio-temporal bounding boxes.
        """
        detected_trajectories = []
        lead_time_snapshots = []
        
        # Evaluate each forecast horizon step (Day 3 through Day 10)
        for idx, day in enumerate(lead_days):
            u_field = ensemble_forecast_4d["u_wind"][idx]
            v_field = ensemble_forecast_4d["v_wind"][idx]
            t_field = ensemble_forecast_4d["t2m"][idx]
            p_field = ensemble_forecast_4d["mslp"][idx]
            precip_field = ensemble_forecast_4d["precip"][idx]
            q_field = ensemble_forecast_4d["humidity"][idx]
            z_field = ensemble_forecast_4d["z500"][idx]
            
            # Wind speed magnitude
            wind_speed = np.sqrt(u_field**2 + v_field**2)
            
            # Map regular field to spherical mesh
            node_u = self.mesh.project_grid_to_mesh(ensemble_forecast_4d["lats"], ensemble_forecast_4d["lons"], u_field)
            node_v = self.mesh.project_grid_to_mesh(ensemble_forecast_4d["lats"], ensemble_forecast_4d["lons"], v_field)
            node_t = self.mesh.project_grid_to_mesh(ensemble_forecast_4d["lats"], ensemble_forecast_4d["lons"], t_field)
            node_p = self.mesh.project_grid_to_mesh(ensemble_forecast_4d["lats"], ensemble_forecast_4d["lons"], p_field)
            node_pr = self.mesh.project_grid_to_mesh(ensemble_forecast_4d["lats"], ensemble_forecast_4d["lons"], precip_field)
            node_q = self.mesh.project_grid_to_mesh(ensemble_forecast_4d["lats"], ensemble_forecast_4d["lons"], q_field)
            node_z = self.mesh.project_grid_to_mesh(ensemble_forecast_4d["lats"], ensemble_forecast_4d["lons"], z_field)
            
            # Form tensor [num_nodes, 7]
            node_feats = np.column_stack([node_u, node_v, node_t, node_p, node_pr, node_q, node_z])
            t_feats = torch.tensor(node_feats, dtype=torch.float32)
            
            self.eval()
            with torch.no_grad():
                prob, efi = self.forward(t_feats)
                prob_np = prob.squeeze().cpu().numpy()
                efi_np = efi.squeeze().cpu().numpy()
            
            # Identify severe anomalies: efi > 0.65 or anomaly prob > 0.70
            severe_mask = (efi_np > 0.60) | (prob_np > 0.65)
            severe_indices = np.where(severe_mask)[0]
            
            if len(severe_indices) > 0:
                cluster_lats = self.mesh.lats[severe_indices]
                cluster_lons = self.mesh.lons[severe_indices]
                cluster_efi = efi_np[severe_indices]
                
                # Weighted centroid by EFI
                weights = cluster_efi - np.min(cluster_efi) + 0.1
                centroid_lat = float(np.average(cluster_lats, weights=weights))
                centroid_lon = float(np.average(cluster_lons, weights=weights))
                max_efi_val = float(np.max(cluster_efi))
                
                # Find peak atmospheric variables at centroid
                lat_idx = np.argmin(np.abs(ensemble_forecast_4d["lats"][:, 0] - centroid_lat))
                lon_idx = np.argmin(np.abs(ensemble_forecast_4d["lons"][0, :] - centroid_lon))
                
                max_wind = float(wind_speed[lat_idx, lon_idx] * 3.6) # km/h
                min_mslp = float(p_field[lat_idx, lon_idx])
                peak_precip = float(precip_field[lat_idx, lon_idx])
                temp_val = float(t_field[lat_idx, lon_idx] - 273.15) # Celsius
                
                lead_time_snapshots.append({
                    "step_id": idx,
                    "lead_day": float(day),
                    "valid_time": (datetime.utcnow() + timedelta(days=float(day))).isoformat(),
                    "lat": round(centroid_lat, 4),
                    "lon": round(centroid_lon, 4),
                    "max_wind_kmh": round(max_wind, 1),
                    "min_mslp_hpa": round(min_mslp, 1),
                    "peak_precip_mmh": round(peak_precip, 2),
                    "temperature_c": round(temp_val, 1),
                    "efi_score": round(max_efi_val, 3),
                    "uncertainty_spread": round(float(0.08 * (day / 3.0)), 3)
                })

        # Assemble full trajectory & 4D bounding box if waypoints found
        if lead_time_snapshots:
            all_lats = [pt["lat"] for pt in lead_time_snapshots]
            all_lons = [pt["lon"] for pt in lead_time_snapshots]
            
            # Determine anomaly class
            peak_wind = max(pt["max_wind_kmh"] for pt in lead_time_snapshots)
            peak_pr = max(pt["peak_precip_mmh"] for pt in lead_time_snapshots)
            peak_temp = max(pt["temperature_c"] for pt in lead_time_snapshots)
            
            if peak_wind > 80.0:
                event_type = "CYCLONE"
                name = "Severe Cyclonic Storm System"
            elif peak_temp > 43.0:
                event_type = "HEATWAVE"
                name = "Severe Synoptic Heat Dome"
            elif peak_pr > 50.0:
                event_type = "EXTREME_PRECIPITATION"
                name = "Mesoscale Cloudburst / Extreme Downpour"
            else:
                event_type = "SEVERE_CONVECTIVE_STORM"
                name = "Severe Atmospheric Disturbance"
                
            max_efi = max(pt["efi_score"] for pt in lead_time_snapshots)
            severity = "SEVERE" if max_efi > 0.8 else ("MODERATE" if max_efi > 0.6 else "LOW")
            
            # 4D Macro-scale bounding box (+/- 3 degrees padding for diffusion crop)
            pad = 3.5
            bbox_4d = {
                "lead_start_day": min(pt["lead_day"] for pt in lead_time_snapshots),
                "lead_end_day": max(pt["lead_day"] for pt in lead_time_snapshots),
                "lat_min": round(max(-90.0, min(all_lats) - pad), 3),
                "lat_max": round(min(90.0, max(all_lats) + pad), 3),
                "lon_min": round(max(-180.0, min(all_lons) - pad), 3),
                "lon_max": round(min(180.0, max(all_lons) + pad), 3),
                "pressure_levels_hpa": [1000, 850, 700, 500, 300, 200]
            }
            
            detected_trajectories.append({
                "anomaly_id": f"ANO-{int(datetime.utcnow().timestamp())}",
                "event_type": event_type,
                "name": name,
                "description": f"GNN-isolated 4D trajectory tracking {event_type} anomaly across Day {bbox_4d['lead_start_day']} to Day {bbox_4d['lead_end_day']} with peak EFI {max_efi}.",
                "severity": severity,
                "max_efi": max_efi,
                "bounding_box": bbox_4d,
                "trajectory": lead_time_snapshots
            })
            
        return detected_trajectories
