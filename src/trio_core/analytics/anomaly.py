"""Anomaly detection — z-score deviation from historical patterns.

Compares current counts against same-hour/same-day-of-week baselines.
Flags deviations > 2 standard deviations.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime

from trio_core.analytics.aggregator import Bin


@dataclass
class Anomaly:
    """A detected anomaly in traffic data."""
    timestamp: datetime
    expected: float
    actual: float
    z_score: float
    deviation_pct: float
    direction: str  # "above" or "below"
    description: str
    severity: str  # "low", "medium", "high"


class AnomalyDetector:
    """Detect anomalies by comparing bins against historical baselines."""

    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold

    def detect(self, bins: list[Bin], historical_bins: list[Bin] | None = None) -> list[Anomaly]:
        """Detect anomalies in bins.

        If historical_bins provided, compare against same-hour/same-dow baselines.
        Otherwise, use the bins themselves to compute internal anomalies.
        """
        if not bins:
            return []

        if historical_bins:
            return self._detect_vs_historical(bins, historical_bins)
        return self._detect_internal(bins)

    def _detect_internal(self, bins: list[Bin]) -> list[Anomaly]:
        """Detect anomalies within a single set of bins using global stats."""
        if len(bins) < 4:
            return []

        counts = [b.count for b in bins]
        mean = statistics.mean(counts)
        std = statistics.stdev(counts) if len(counts) > 1 else 0

        if std == 0 or mean == 0:
            return []

        anomalies = []
        for b in bins:
            z = (b.count - mean) / std
            if abs(z) > self.z_threshold:
                deviation_pct = abs(b.count - mean) / mean * 100
                direction = "above" if b.count > mean else "below"
                severity = self._severity(abs(z))
                anomalies.append(Anomaly(
                    timestamp=b.start,
                    expected=round(mean, 1),
                    actual=b.count,
                    z_score=round(z, 2),
                    deviation_pct=round(deviation_pct, 1),
                    direction=direction,
                    description=f"Traffic {deviation_pct:.0f}% {direction} average at {b.start.strftime('%H:%M')} (expected ~{mean:.0f}, got {b.count})",
                    severity=severity,
                ))

        return anomalies

    def _detect_vs_historical(self, bins: list[Bin], historical: list[Bin]) -> list[Anomaly]:
        """Compare current bins against historical same-hour/same-dow."""
        # Build baseline: group historical by (dow, hour)
        baselines: dict[str, list[float]] = {}
        for b in historical:
            key = f"{b.start.weekday()}_{b.start.hour:02d}"
            baselines.setdefault(key, []).append(b.count)

        anomalies = []
        for b in bins:
            key = f"{b.start.weekday()}_{b.start.hour:02d}"
            historical_counts = baselines.get(key, [])

            if len(historical_counts) < 2:
                continue

            mean = statistics.mean(historical_counts)
            std = statistics.stdev(historical_counts)

            if std == 0 or mean == 0:
                continue

            z = (b.count - mean) / std
            if abs(z) > self.z_threshold:
                deviation_pct = abs(b.count - mean) / mean * 100
                direction = "above" if b.count > mean else "below"
                severity = self._severity(abs(z))
                dow_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][b.start.weekday()]
                anomalies.append(Anomaly(
                    timestamp=b.start,
                    expected=round(mean, 1),
                    actual=b.count,
                    z_score=round(z, 2),
                    deviation_pct=round(deviation_pct, 1),
                    direction=direction,
                    description=f"Traffic {deviation_pct:.0f}% {direction} typical {dow_name} {b.start.hour:02d}:00 (expected ~{mean:.0f}, got {b.count})",
                    severity=severity,
                ))

        return anomalies

    @staticmethod
    def _severity(abs_z: float) -> str:
        if abs_z > 3.0:
            return "high"
        if abs_z > 2.5:
            return "medium"
        return "low"
