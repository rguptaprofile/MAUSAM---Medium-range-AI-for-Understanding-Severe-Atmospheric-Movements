"""
Unit tests for IcosahedralMesh, GNN Message Passing, and EFI Tracking.
"""
import unittest
import numpy as np
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from backend.app.core.spherical_mesh import IcosahedralMesh
from backend.app.core.gnn_tracker import SphericalGNNAnomalyTracker, compute_extreme_forecast_index

class TestGNNTracker(unittest.TestCase):
    def test_icosphere_subdivision(self):
        # Level 2 mesh
        mesh2 = IcosahedralMesh(subdivision_level=2)
        # Expected vertices: 10 * 4^2 + 2 = 162
        self.assertEqual(len(mesh2.vertices), 162)
        self.assertEqual(len(mesh2.lats), 162)
        self.assertEqual(mesh2.edge_index.shape[0], 2)

        # Level 3 mesh
        mesh3 = IcosahedralMesh(subdivision_level=3)
        # Expected vertices: 10 * 4^3 + 2 = 642
        self.assertEqual(len(mesh3.vertices), 642)

    def test_efi_computation(self):
        # Historical baseline percentiles
        clim_percentiles = np.linspace(10.0, 50.0, 99)
        
        # Moderate forecast ensemble -> EFI should be near 0
        moderate_forecast = np.linspace(15.0, 45.0, 21)
        efi_mod = compute_extreme_forecast_index(moderate_forecast, clim_percentiles)
        self.assertAlmostEqual(efi_mod, 0.0, delta=0.35)

        # Extreme forecast ensemble far exceeding 99th percentile -> EFI near +1.0
        extreme_forecast = np.full(21, 95.0)
        efi_ext = compute_extreme_forecast_index(extreme_forecast, clim_percentiles)
        self.assertGreater(efi_ext, 0.8)

    def test_gnn_forward_pass(self):
        tracker = SphericalGNNAnomalyTracker(in_features=7, hidden_dim=32, mesh_level=2)
        num_nodes = tracker.mesh.num_nodes
        if HAS_TORCH:
            dummy_feats = torch.randn((num_nodes, 7))
            prob, efi = tracker(dummy_feats)
            self.assertEqual(prob.shape, (num_nodes, 1))
            self.assertEqual(efi.shape, (num_nodes, 1))
            self.assertTrue((prob >= 0.0).all() and (prob <= 1.0).all())
        else:
            self.assertEqual(num_nodes, 162)


if __name__ == "__main__":
    unittest.main()
