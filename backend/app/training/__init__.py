"""
MAUSAM Training & Continual Learning Package (SIH26078).
Exports CanonicalDatasetBuilder, VerificationEngine, and TruthLaggedContinualLearningEngine.
"""
from .dataset_builder import CanonicalDatasetBuilder
from .verification import VerificationEngine, verification_engine
from .continual_learning import TruthLaggedContinualLearningEngine, continual_learning_engine

__all__ = [
    "CanonicalDatasetBuilder",
    "VerificationEngine",
    "verification_engine",
    "TruthLaggedContinualLearningEngine",
    "continual_learning_engine"
]
