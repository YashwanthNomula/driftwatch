"""driftwatch — detect data drift between training and serving data. Zero dependencies."""

from .alerts import (
    Alert,
    AlertHook,
    detect_alerts,
    dispatch_alerts,
    filter_already_reported,
    load_alert_state,
    parse_alert_hook,
    record_alerts,
    save_alert_state,
)
from .config import ConfigError, ThresholdConfig, ThresholdSet, load_config
from .detect import compare_datasets, DriftReport, FeatureResult
from .trend import trend_series, TrendReport, TrendFeature
from . import stats

__all__ = ["compare_datasets", "trend_series", "DriftReport", "FeatureResult",
           "TrendReport", "TrendFeature", "stats",
           "load_config", "ThresholdConfig", "ThresholdSet", "ConfigError",
           "Alert", "AlertHook", "detect_alerts", "dispatch_alerts",
           "filter_already_reported", "load_alert_state", "parse_alert_hook",
           "record_alerts", "save_alert_state"]

__version__ = "1.3.0"
