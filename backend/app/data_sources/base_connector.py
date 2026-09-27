"""
Base Data Source Connector for MAUSAM SIH26078.
Defines unified interface for operational NWP, reanalysis, and observational feeds.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import hashlib
import os
import logging
from datetime import datetime

logger = logging.getLogger("mausam.data_sources.base")

class BaseDataSourceConnector(ABC):
    def __init__(self, source_id: str, provider: str, resolution_km: float, is_primary: bool = False):
        self.source_id = source_id
        self.provider = provider
        self.resolution_km = resolution_km
        self.is_primary = is_primary
        self.last_successful_cycle: Optional[str] = None
        self.last_retrieval_time: Optional[datetime] = None
        self.latency_seconds: float = 0.0
        self.quality_status: str = "ONLINE" # ONLINE, DEGRADED, OFFLINE

    @abstractmethod
    def fetch_cycle(self, cycle_id: Optional[str] = None, lead_days: Optional[List[float]] = None) -> Dict[str, Any]:
        """Fetches and ingests a specific cycle or the latest operational cycle."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Checks API reachability, storage availability, and latency."""
        pass

    def compute_sha256(self, file_path_or_bytes: Any) -> str:
        """Computes verifiable cryptographic SHA-256 for data provenance."""
        hasher = hashlib.sha256()
        if isinstance(file_path_or_bytes, (bytes, bytearray)):
            hasher.update(file_path_or_bytes)
        elif isinstance(file_path_or_bytes, str) and os.path.exists(file_path_or_bytes):
            with open(file_path_or_bytes, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        else:
            hasher.update(str(file_path_or_bytes).encode("utf-8"))
        return hasher.hexdigest()

    def get_source_card(self, cycle_id: str, checksum: str, valid_time_range: str, file_id: str) -> Dict[str, Any]:
        """Emits standard SIH26078 data source card for UI and audit logs."""
        return {
            "source_name": self.source_id,
            "provider": self.provider,
            "forecast_cycle": cycle_id,
            "init_time": datetime.utcnow().isoformat(),
            "valid_time_range": valid_time_range,
            "file_id": file_id,
            "checksum": checksum,
            "retrieval_time": datetime.utcnow().isoformat(),
            "quality_status": self.quality_status,
            "resolution_km": self.resolution_km,
            "is_primary": self.is_primary
        }
