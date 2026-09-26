"""
Unit tests for AtmosphericPhysicsLoss in MAUSAM.
"""
import unittest
import torch
from backend.app.core.physics_loss import AtmosphericPhysicsLoss

class TestAtmosphericPhysicsLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = AtmosphericPhysicsLoss(grid_spacing_km=5.0)

    def test_spatial_gradients(self):
        # Create known 2D ramp field
        field = torch.ones((1, 1, 10, 10))
        df_dx, df_dy = self.loss_fn.compute_spatial_gradients(field)
        self.assertEqual(df_dx.shape, field.shape)
        self.assertEqual(df_dy.shape, field.shape)
        # Uniform field should have zero gradient
        self.assertTrue(torch.allclose(df_dx, torch.zeros_like(df_dx)))
        self.assertTrue(torch.allclose(df_dy, torch.zeros_like(df_dy)))

    def test_non_negativity_loss(self):
        # Strictly positive precip & humidity should give zero penalty
        pos_precip = torch.tensor([[10.0, 5.0], [2.0, 1.0]])
        pos_q = torch.tensor([[0.015, 0.012], [0.010, 0.014]])
        loss_zero = self.loss_fn.non_negativity_loss(pos_precip, pos_q)
        self.assertEqual(loss_zero.item(), 0.0)

        # Negative precip should be severely penalized
        neg_precip = torch.tensor([[-5.0, 2.0], [0.0, -10.0]])
        loss_penalized = self.loss_fn.non_negativity_loss(neg_precip, pos_q)
        self.assertGreater(loss_penalized.item(), 0.0)

    def test_moisture_continuity_loss(self):
        # Extreme downpour without any moisture flux convergence should incur penalty
        fake_precip = torch.ones((1, 1, 16, 16)) * 100.0 # 100 mm/h
        zero_u = torch.zeros((1, 1, 16, 16))
        zero_v = torch.zeros((1, 1, 16, 16))
        zero_q = torch.zeros((1, 1, 16, 16))
        loss = self.loss_fn.moisture_continuity_loss(fake_precip, zero_u, zero_v, zero_q)
        self.assertGreater(loss.item(), 0.0)

    def test_total_forward(self):
        p = torch.ones((16, 16)) * 10.0
        u = torch.ones((16, 16)) * 5.0
        v = torch.ones((16, 16)) * 2.0
        q = torch.ones((16, 16)) * 0.01
        phi = torch.ones((16, 16)) * 5800.0
        lats = torch.linspace(15.0, 25.0, 16)
        
        report = self.loss_fn(p, u, v, q, phi, lats)
        self.assertIn("total_physics_loss", report)
        self.assertIn("moisture_continuity_loss", report)
        self.assertIn("geostrophic_loss", report)
        self.assertIn("non_negativity_loss", report)

if __name__ == "__main__":
    unittest.main()
