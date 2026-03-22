"""Temporal aggregation — bin raw count samples into 15-min / hourly / daily buckets.

Design based on exp3 analysis:
- Use median (robust to outliers) within each bin
- 15-min bins are the base unit (industry standard: RetailNext, Sensormatic)
- Hourly = sum of 4 x 15-min bins
- Provides confidence, velocity, min/max per bin
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal


@dataclass
class Sample:
    """A single count measurement from the counter pipeline."""
    timestamp: datetime
    count: int          # corrected person count
    raw_count: int      # raw YOLO detections
    velocity: float = 0.0
    confidence: float = 1.0
    camera_id: str = ""


@dataclass
class Bin:
    """An aggregated time bin (15-min, hourly, or daily)."""
    start: datetime
    end: datetime
    count: int              # median of samples (for 15-min) or sum of sub-bins
    mean: float = 0.0
    median: float = 0.0
    min_count: int = 0
    max_count: int = 0
    samples: int = 0        # number of raw measurements in this bin
    confidence: float = 1.0 # weighted avg confidence
    velocity: float = 0.0   # avg velocity (trend direction)
    camera_id: str = ""

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60


class Aggregator:
    """Aggregate raw count samples into temporal bins.

    Uses mean aggregation by default (optimal when input is Kalman-smoothed).
    Median available for raw/unfiltered input.
    """

    def __init__(self, bin_minutes: int = 15, agg_method: Literal["mean", "median"] = "mean"):
        self.bin_minutes = bin_minutes
        self.agg_method = agg_method

    def aggregate(
        self,
        samples: list[Sample],
        level: Literal["bin", "hourly", "daily"] = "bin",
    ) -> list[Bin]:
        """Aggregate samples into bins at the specified level."""
        if not samples:
            return []

        # Sort by timestamp
        samples = sorted(samples, key=lambda s: s.timestamp)

        # Step 1: Base bins (default 15-min)
        base_bins = self._bin_samples(samples, self.bin_minutes)

        if level == "bin":
            return base_bins

        # Step 2: Hourly = group base bins by hour, sum counts
        if level == "hourly":
            return self._roll_up(base_bins, 60)

        # Step 3: Daily = group hourly by day, sum counts
        if level == "daily":
            hourly = self._roll_up(base_bins, 60)
            return self._roll_up(hourly, 1440)

        return base_bins

    def _bin_samples(self, samples: list[Sample], minutes: int) -> list[Bin]:
        """Group samples into fixed-width time bins using median."""
        if not samples:
            return []

        bins: list[Bin] = []
        delta = timedelta(minutes=minutes)

        # Find bin boundaries
        first_ts = samples[0].timestamp
        # Align to bin boundary
        bin_start = first_ts.replace(
            minute=(first_ts.minute // minutes) * minutes,
            second=0, microsecond=0,
        )

        current_bin_samples: list[Sample] = []
        bin_end = bin_start + delta

        for s in samples:
            while s.timestamp >= bin_end:
                # Flush current bin
                if current_bin_samples:
                    bins.append(self._make_bin(current_bin_samples, bin_start, bin_end))
                current_bin_samples = []
                bin_start = bin_end
                bin_end = bin_start + delta

            current_bin_samples.append(s)

        # Flush last bin
        if current_bin_samples:
            bins.append(self._make_bin(current_bin_samples, bin_start, bin_end))

        return bins

    def _make_bin(self, samples: list[Sample], start: datetime, end: datetime) -> Bin:
        """Create a Bin from a list of samples."""
        counts = [s.count for s in samples]
        velocities = [s.velocity for s in samples]
        confidences = [s.confidence for s in samples]

        median_count = statistics.median(counts)
        mean_count = statistics.mean(counts)
        agg_count = mean_count if self.agg_method == "mean" else median_count

        return Bin(
            start=start,
            end=end,
            count=round(agg_count),
            mean=round(mean_count, 1),
            median=round(median_count, 1),
            min_count=min(counts),
            max_count=max(counts),
            samples=len(samples),
            confidence=round(statistics.mean(confidences), 3),
            velocity=round(statistics.mean(velocities), 3),
            camera_id=samples[0].camera_id if samples else "",
        )

    def _roll_up(self, bins: list[Bin], minutes: int) -> list[Bin]:
        """Roll up bins into larger windows by summing counts."""
        if not bins:
            return []

        delta = timedelta(minutes=minutes)
        result: list[Bin] = []

        # Group bins by parent window
        first_start = bins[0].start
        window_start = first_start.replace(
            hour=first_start.hour if minutes < 1440 else 0,
            minute=0, second=0, microsecond=0,
        )
        if minutes >= 1440:
            window_start = window_start.replace(hour=0)

        current_group: list[Bin] = []
        window_end = window_start + delta

        for b in bins:
            while b.start >= window_end:
                if current_group:
                    result.append(self._sum_bins(current_group, window_start, window_end))
                current_group = []
                window_start = window_end
                window_end = window_start + delta

            current_group.append(b)

        if current_group:
            result.append(self._sum_bins(current_group, window_start, window_end))

        return result

    def _sum_bins(self, bins: list[Bin], start: datetime, end: datetime) -> Bin:
        """Sum sub-bins into a parent bin."""
        total_count = sum(b.count for b in bins)
        total_samples = sum(b.samples for b in bins)
        all_confidences = [b.confidence for b in bins if b.samples > 0]
        all_velocities = [b.velocity for b in bins if b.samples > 0]

        return Bin(
            start=start,
            end=end,
            count=total_count,
            mean=round(total_count / max(len(bins), 1), 1),
            median=round(statistics.median(b.count for b in bins), 1) if bins else 0,
            min_count=min(b.count for b in bins) if bins else 0,
            max_count=max(b.count for b in bins) if bins else 0,
            samples=total_samples,
            confidence=round(statistics.mean(all_confidences), 3) if all_confidences else 0,
            velocity=round(statistics.mean(all_velocities), 3) if all_velocities else 0,
            camera_id=bins[0].camera_id if bins else "",
        )
