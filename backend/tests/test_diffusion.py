"""
Unit tests for Stage-2 Conditional Diffusion Downscaler in MAUSAM.
"""
import unittest
import numpy as np
from backend.app.core.diffusion_downscaler import ConditionalDiffusionDownscaler

class TestDiffusionDownscaler(unittest.TestCase):
    def setUp(self):
        self.downscaler = ConditionalDiffusionDownscaler(num_timesteps=5) # fast steps for test

    def test_downscaling_and_amplitude_preservation(self):
        # Create a synthetic coarse 12km slice with a severe peak
        slice_12km = np.zeros((16, 16), dtype=np.float32)
        slice_12km[8, 8] = 75.0 # 75 mm/h peak
        
        result = self.downscaler.downscale_anomaly_slice(
            slice_12km,
            target_shape=(32, 32),
            variable_type="precipitation"
        )
        
        self.assertIn("downscaled_5km_grid", result)
        self.assertIn("peak_amplitude_coarse", result)
        self.assertIn("peak_amplitude_downscaled", result)
        self.assertIn("peak_amplitude_cnn_smoothed", result)
        
        # Verify that diffusion retains higher peak amplitude than smoothed CNN
        peak_diff = result["peak_amplitude_downscaled"]
        peak_cnn = result["peak_amplitude_cnn_smoothed"]
        self.assertGreater(peak_diff, peak_cnn)
        
        # Verify grid dimensions
        grid_5km = np.array(result["downscaled_5km_grid"])
        self.assertEqual(grid_5km.shape, (32, 32))

if __name__ == "__main__":
    unittest.main()
