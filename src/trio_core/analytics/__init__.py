"""Temporal analytics — aggregation, anomaly detection, pattern extraction."""

from trio_core.analytics.aggregator import Aggregator, Bin
from trio_core.analytics.anomaly import AnomalyDetector, Anomaly

__all__ = ["Aggregator", "Bin", "AnomalyDetector", "Anomaly"]
