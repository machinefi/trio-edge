"""Temporal analytics — aggregation, anomaly detection, pattern extraction."""

from trio_core.analytics.aggregator import Aggregator, Bin
from trio_core.analytics.anomaly import Anomaly, AnomalyDetector

__all__ = ["Aggregator", "Bin", "AnomalyDetector", "Anomaly"]
