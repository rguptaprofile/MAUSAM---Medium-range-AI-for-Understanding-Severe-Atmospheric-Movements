"""
Unit tests for AtmosphericPhysicsLoss in MAUSAM SIH26078.
"""
import unittest
import numpy as np
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from backend.app.core.physics_loss import AtmosphericPhysicsLoss

class TestAtmosphericPhysicsLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = AtmosphericPhysicsLoss(grid_spacing_km=5.0)

    def test_spatial_gradients(self):
        if HAS_TORCH:
            field = torch.ones((1, 1, 10, 10))
            df_dx, df_dy = self.loss_fn.compute_spatial_gradients(field)
            self.assertEqual(df_dx.shape, field.shape)
            self.assertEqual(df_dy.shape, field.shape)
            self.assertTrue(torch.allclose(df_dx, torch.zeros_like(df_dx)))
        else:
            field = np.ones((10, 10), dtype=np.float32)
            df_dx, df_dy = self.loss_fn.compute_spatial_gradients(field)
            self.assertEqual(df_dx.shape, (10, 10))
            self.assertTrue(np.allclose(df_dx, 0.0))

    def test_non_negativity_loss(self):
        if HAS_TORCH:
            pos_precip = torch.tensor([[10.0, 5.0], [2.0, 1.0]])
            pos_q = torch.tensor([[0.015, 0.012], [0.010, 0.014]])
            loss_zero = self.loss_fn.non_negativity_loss(pos_precip, pos_q)
            self.assertEqual(loss_zero.item(), 0.0)

            neg_precip = torch.tensor([[-5.0, 2.0], [0.0, -10.0]])
            loss_penalized = self.loss_fn.non_negativity_loss(neg_precip, pos_q)
            self.assertGreater(loss_penalized.item(), 0.0)
        else:
            pos_precip = np.array([[10.0, 5.0], [2.0, 1.0]])
            pos_q = np.array([[0.015, 0.012], [0.010, 0.014]])
            loss_zero = self.loss_fn.non_negativity_loss(pos_precip, pos_q)
            self.assertEqual(float(loss_zero), 0.0)

            neg_precip = np.array([[-5.0, 2.0], [0.0, -10.0]])
            loss_penalized = self.loss_fn.non_negativity_loss(neg_precip, pos_q)
            self.assertGreater(float(loss_penalized), 0.0)

    def test_moisture_continuity_loss(self):
        if HAS_TORCH:
            fake_precip = torch.ones((1, 1, 16, 16)) * 100.0
            zero_u = torch.zeros((1, 1, 16, 16))
            zero_v = torch.zeros((1, 1, 16, 16))
            zero_q = torch.zeros((1, 1, 16, 16))
            loss = self.loss_fn.moisture_continuity_loss(fake_precip, zero_u, zero_v, zero_q)
            self.assertGreater(loss.item(), 0.0)
        else:
            fake_precip = np.ones((16, 16)) * 100.0
            zero_u = np.zeros((16, 16))
            zero_v = np.zeros((16, 16))
            zero_q = np.zeros((16, 16))
            loss = self.loss_fn.moisture_continuity_loss(fake_precip, zero_u, zero_v, zero_q)
            self.assertGreater(float(loss), 0.0)

    def test_total_forward(self):
        if HAS_TORCH:
            p = torch.ones((16, 16)) * 10.0
            u = torch.ones((16, 16)) * 5.0
            v = torch.ones((16, 16)) * 2.0
            q = torch.ones((16, 16)) * 0.01
            phi = torch.ones((16, 16)) * 5800.0
            lats = torch.linspace(15.0, 25.0, 16)
            report = self.loss_fn(p, u, v, q, phi, lats)
            self.assertIn("total_physics_loss", report)
        else:
            p = np.ones((16, 16)) * 10.0
            u = np.ones((16, 16)) * 5.0
            v = np.ones((16, 16)) * 2.0
            q = np.ones((16, 16)) * 0.01
            phi = np.ones((16, 16)) * 5800.0
            lats = np.linspace(15.0, 25.0, 16)
            report = self.loss_fn(p, u, v, q, phi, lats)
            self.assertIn("total_physics_loss", report)

if __name__ == "__main__":
    unittest.main()
