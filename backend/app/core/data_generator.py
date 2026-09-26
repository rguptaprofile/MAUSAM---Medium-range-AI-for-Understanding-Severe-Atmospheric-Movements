"""
Meteorological Data Generator & Climatology Synthesizer.
Simulates 12 km NCMRWF Global Ensemble (NEPS-G) 4D data arrays
and 30-year ERA5/IMDAA climatological baselines for historical benchmark scenarios.
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

class SyntheticMeteorologicalDataGenerator:
    def __init__(self, lat_range=(5.0, 35.0), lon_range=(65.0, 95.0), res_deg=0.12):
        """
        Covers the Indian Subcontinent and adjoining Indian Ocean, Arabian Sea, and Bay of Bengal.
        12 km resolution roughly corresponds to ~0.12 degrees.
        """
        self.lats_1d = np.arange(lat_range[0], lat_range[1] + res_deg, res_deg)
        self.lons_1d = np.arange(lon_range[0], lon_range[1] + res_deg, res_deg)
        self.LATS, self.LONS = np.meshgrid(self.lats_1d, self.lons_1d, indexing='ij')
        self.num_lats, self.num_lons = self.LATS.shape

    def generate_benchmark_scenario(self, scenario_type: str = "cyclone_amphan", lead_days: List[float] = None) -> Dict[str, Any]:
        """
        Generates 4D ensemble fields for benchmark extreme weather events.
        Supported: 'cyclone_amphan', 'north_india_heatwave', 'monsoon_cloudburst'
        """
        if lead_days is None:
            lead_days = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            
        num_steps = len(lead_days)
        shape_4d = (num_steps, self.num_lats, self.num_lons)
        
        # Base ambient fields
        u_wind = np.random.normal(5.0, 2.0, size=shape_4d).astype(np.float32)
        v_wind = np.random.normal(3.0, 2.0, size=shape_4d).astype(np.float32)
        t2m = np.random.normal(303.15, 3.0, size=shape_4d).astype(np.float32) # Kelvin (~30 C)
        mslp = np.random.normal(1010.0, 2.0, size=shape_4d).astype(np.float32) # hPa
        precip = np.random.exponential(1.5, size=shape_4d).astype(np.float32) # mm/h
        humidity = np.random.uniform(0.010, 0.018, size=shape_4d).astype(np.float32) # kg/kg
        z500 = np.random.normal(5820.0, 30.0, size=shape_4d).astype(np.float32) # gpm

        if scenario_type == "cyclone_amphan":
            # Cyclone tracking north-north-east across Bay of Bengal toward West Bengal/Odisha
            base_lat, base_lon = 12.0, 86.5
            lat_drift = 2.1
            lon_drift = 0.9
            
            for i, day in enumerate(lead_days):
                c_lat = base_lat + (day - 3.0) * lat_drift
                c_lon = base_lon + (day - 3.0) * lon_drift
                
                # Gaussian distance metric from cyclone vortex center
                dist_sq = (self.LATS - c_lat)**2 + (self.LONS - c_lon)**2
                core_radius = 2.0 # degrees
                vortex_decay = np.exp(-dist_sq / (2.0 * core_radius**2))
                
                # Cyclonic rotation: tangent vector (-dlat, dlon)
                tangent_u = - (self.LATS - c_lat)
                tangent_v = (self.LONS - c_lon)
                norm = np.sqrt(tangent_u**2 + tangent_v**2) + 1e-5
                
                intensity_scale = max(0.0, 1.0 - abs(day - 6.0) * 0.12) # peaks around Day 5-6
                cyclonic_speed = (45.0 * intensity_scale) * vortex_decay # m/s (~160 km/h)
                
                u_wind[i] += (tangent_u / norm) * cyclonic_speed
                v_wind[i] += (tangent_v / norm) * cyclonic_speed
                mslp[i] -= (85.0 * intensity_scale) * vortex_decay # drops down to ~925 hPa
                precip[i] += (65.0 * intensity_scale) * vortex_decay # up to 70+ mm/h
                humidity[i] += 0.010 * vortex_decay
                z500[i] -= (180.0 * intensity_scale) * vortex_decay

        elif scenario_type == "north_india_heatwave":
            # Severe heat dome centered over Rajasthan, Delhi NCR, and UP
            center_lat, center_lon = 28.5, 76.5
            for i, day in enumerate(lead_days):
                dist_sq = (self.LATS - center_lat)**2 + (self.LONS - center_lon)**2
                dome_decay = np.exp(-dist_sq / 18.0)
                
                # Anticyclonic high pressure dome & intense 2m temperature
                t2m[i] += (17.5 * dome_decay) # reaches 46-48 C (320 K)
                mslp[i] += (8.0 * dome_decay)
                z500[i] += (160.0 * dome_decay) # blocking ridge
                precip[i] = np.maximum(0.0, precip[i] * (1.0 - dome_decay))
                humidity[i] = np.maximum(0.003, humidity[i] * (1.0 - 0.7 * dome_decay))

        elif scenario_type == "monsoon_cloudburst":
            # Mesoscale extreme convective downpour along the Western Ghats / Konkan coast
            center_lat, center_lon = 19.0, 72.8
            for i, day in enumerate(lead_days):
                dist_sq = (self.LATS - center_lat)**2 + (self.LONS - center_lon)**2
                burst_decay = np.exp(-dist_sq / 1.5)
                
                # Heavy orographic moisture convergence & cloudburst
                precip[i] += 88.0 * burst_decay # 90+ mm/h intense cloudburst
                u_wind[i] += 22.0 * burst_decay # strong westerly monsoon surge
                humidity[i] += 0.012 * burst_decay
                mslp[i] -= 18.0 * burst_decay

        elif scenario_type == "north_india_coldwave":
            # Severe winter cold wave & ground frost anomaly over North/North-West India
            center_lat, center_lon = 29.5, 75.8
            for i, day in enumerate(lead_days):
                dist_sq = (self.LATS - center_lat)**2 + (self.LONS - center_lon)**2
                cold_decay = np.exp(-dist_sq / 16.0)
                
                # Arctic northerly continental surge: severe temperature plunge to ~2-4°C
                t2m[i] -= (26.0 * cold_decay) # drops from 303K down to 275-277K (2-4 C)
                mslp[i] += (14.0 * cold_decay) # intense winter high-pressure ridge (~1024 hPa)
                v_wind[i] -= (18.0 * cold_decay) # strong northerly icy winds
                u_wind[i] += (8.0 * cold_decay)
                humidity[i] = np.maximum(0.002, humidity[i] * (1.0 - 0.75 * cold_decay)) # dry cold air
                precip[i] = np.zeros_like(precip[i]) # clear frosty sky

        # Compute climatological percentile baseline (30-year ERA5 approximation)
        climatology_baseline = {
            "p50_precip": 1.2,
            "p90_precip": 12.0,
            "p99_precip": 35.0,
            "p50_wind": 4.5,
            "p90_wind": 15.0,
            "p99_wind": 28.0,
            "p50_t2m": 302.0,
            "p99_t2m": 315.0
        }

        return {
            "scenario": scenario_type,
            "lead_days": lead_days,
            "lats": self.LATS,
            "lons": self.LONS,
            "u_wind": u_wind,
            "v_wind": v_wind,
            "t2m": t2m,
            "mslp": mslp,
            "precip": precip,
            "humidity": humidity,
            "z500": z500,
            "climatology_baseline": climatology_baseline
        }
