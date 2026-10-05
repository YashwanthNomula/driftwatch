"""driftwatch — detect data drift between training and serving data. Zero dependencies."""

from .detect import compare_datasets, DriftReport, FeatureResult
from . import stats

__all__ = ["compare_datasets", "DriftReport", "FeatureResult", "stats"]
__version__ = "1.0.0"
