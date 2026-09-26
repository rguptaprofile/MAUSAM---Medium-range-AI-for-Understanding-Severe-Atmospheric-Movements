"""
Integration tests for MAUSAM REST API.
"""
import unittest
from fastapi.testclient import TestClient
from backend.app.main import app as main_app
from backend.app.database.mongo import db

class TestMausamAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main_app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("database", data)

    def test_analytics_endpoints(self):
        res_phys = self.client.get("/api/analytics/physics-metrics")
        self.assertEqual(res_phys.status_code, 200)
        self.assertIn("thermodynamic_compliance_percentage", res_phys.json())

        res_spec = self.client.get("/api/analytics/spectral-smoothing-benchmark")
        self.assertEqual(res_spec.status_code, 200)
        self.assertIn("benchmark_results", res_spec.json())

    def test_alerts_endpoints(self):
        res_alerts = self.client.get("/api/alerts/active")
        self.assertEqual(res_alerts.status_code, 200)
        self.assertIsInstance(res_alerts.json(), list)

        res_geo = self.client.get("/api/alerts/geojson")
        self.assertEqual(res_geo.status_code, 200)
        data = res_geo.json()
        self.assertEqual(data["type"], "FeatureCollection")

if __name__ == "__main__":
    unittest.main()
