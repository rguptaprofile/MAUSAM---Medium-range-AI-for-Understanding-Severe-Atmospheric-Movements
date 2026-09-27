"""
End-to-End Operational Ingestion Coordinator for MAUSAM SIH26078.
Coordinates cycle acquisition, QC validation, unit normalization,
Dask chunking, climatology percentiles alignment, and provenance tagging.
Supports automated rolling ingestion of current live cycle and past 3-10 days cycles.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from ..data_sources import neps_g_source, ncum_g_source, era5_baseline_source, imdaa_source
from .qc_validator import MeteorologicalQCValidator, DataValidationError
from .normalizer import MeteorologicalNormalizer
from .provenance import generate_provenance_record

logger = logging.getLogger("mausam.data_pipeline.ingestion")

class AtmosphericIngestionPipeline:
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.qc_validator = MeteorologicalQCValidator(demo_mode=demo_mode)
        self.normalizer = MeteorologicalNormalizer()

    def ingest_operational_cycle(
        self,
        cycle_id: Optional[str] = None,
        lead_days: Optional[List[float]] = None,
        use_dask: bool = True
    ) -> Dict[str, Any]:
        """
        Ingests and validates an operational NEPS-G forecast cycle:
        1. Queries NCMRWF NEPS-G connector for multi-member 4D fields.
        2. Validates against strict meteorological QC rules.
        3. Standardizes physical units.
        4. Applies Dask-backed chunking.
        5. Attaches ERA5 30-year climatology baseline.
        6. Injects cryptographic provenance card.
        """
        lead_days = lead_days or [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        logger.info(f"Ingesting operational NWP cycle: {cycle_id or 'LATEST_LIVE'}")

        # 1. Fetch raw cycle from NEPS-G connector
        raw_cycle = neps_g_source.fetch_cycle(cycle_id=cycle_id, lead_days=lead_days)

        # 2. Strict QC Validation
        qc_report = self.qc_validator.validate_forecast_dataset(raw_cycle)

        # 3. Unit Normalization
        norm_cycle = self.normalizer.normalize_units(raw_cycle)

        # 4. Optional Dask Chunking for large arrays
        if use_dask:
            norm_cycle = self.normalizer.dask_chunk_arrays(norm_cycle)

        # 5. Fetch Climatology Percentiles for EFI derivation
        H, W = norm_cycle["lats"].shape
        climo_precip = era5_baseline_source.get_climatology_percentiles("precip", shape=(H, W))
        climo_wind = era5_baseline_source.get_climatology_percentiles("wind", shape=(H, W))
        climo_t2m = era5_baseline_source.get_climatology_percentiles("t2m", shape=(H, W))

        # 6. Generate Provenance
        mode = "DEMO" if self.demo_mode else "LIVE"
        provenance = generate_provenance_record(
            source_name=neps_g_source.source_id,
            forecast_cycle=norm_cycle["forecast_cycle"],
            valid_time_range=f"T+{int(lead_days[0]*24)}h to T+{int(lead_days[-1]*24)}h",
            mode=mode,
            uncertainty_level=0.10
        )

        return {
            "status": "INGESTED",
            "cycle_id": norm_cycle["forecast_cycle"],
            "ensemble_members": norm_cycle["ensemble_members"],
            "lead_days": lead_days,
            "lats": norm_cycle["lats"],
            "lons": norm_cycle["lons"],
            "forecast_fields": norm_cycle,
            "climatology": {
                "precip_percentiles": climo_precip,
                "wind_percentiles": climo_wind,
                "t2m_percentiles": climo_t2m,
                "baseline_version": era5_baseline_source.baseline_version
            },
            "qc_report": qc_report,
            "source_card": raw_cycle["source_card"],
            "provenance": provenance
        }

    def ingest_past_rolling_cycles(self, days_back: int = 10) -> List[Dict[str, Any]]:
        """
        Ingests historical forecast cycles from previous 3 to 10 days
        to populate the truth-lagged training store with verified pairs.
        """
        ingested_cycles = []
        now = datetime.utcnow()
        for d in range(3, days_back + 1):
            past_date = now - timedelta(days=d)
            cycle_id = f"NEPSG_{past_date.strftime('%Y%m%d')}_00Z"
            try:
                cycle_data = self.ingest_operational_cycle(cycle_id=cycle_id, use_dask=False)
                ingested_cycles.append(cycle_data)
            except Exception as e:
                logger.warning(f"Could not ingest historical cycle {cycle_id}: {e}")
        return ingested_cycles
