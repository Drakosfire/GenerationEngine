"""Metrics service for tracking generation metrics across calls.

This is a minimal stub - full implementation can be added later.
"""

import logging
from typing import Any

from generationengine.models.metrics import GenerationMetrics

logger = logging.getLogger(__name__)


class MetricsService:
    """
    Service for aggregating and tracking generation metrics.
    
    This is a minimal implementation - can be extended to:
    - Store metrics in database
    - Export to Prometheus/Datadog
    - Calculate aggregates
    """

    def __init__(self):
        self._metrics: list[GenerationMetrics] = []

    def record(self, metrics: GenerationMetrics, service_name: str | None = None) -> None:
        """
        Record a generation metrics object.

        Args:
            metrics: The metrics to record
            service_name: Optional service name for categorization
        """
        self._metrics.append(metrics)
        logger.debug(f"📊 [MetricsService] Recorded metrics for {service_name or 'unknown'}: "
                    f"duration={metrics.duration_ms}ms, cost=${metrics.estimated_cost_usd or 0:.4f}")

    def get_all(self) -> list[GenerationMetrics]:
        """Get all recorded metrics."""
        return self._metrics.copy()

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._metrics.clear()

    def summary(self) -> dict[str, Any]:
        """Get a summary of recorded metrics."""
        if not self._metrics:
            return {
                "count": 0,
                "total_duration_ms": 0,
                "total_cost_usd": 0.0,
                "avg_duration_ms": 0,
            }

        total_duration = sum(m.duration_ms for m in self._metrics)
        total_cost = sum(m.estimated_cost_usd or 0 for m in self._metrics)

        return {
            "count": len(self._metrics),
            "total_duration_ms": total_duration,
            "total_cost_usd": total_cost,
            "avg_duration_ms": total_duration / len(self._metrics),
        }

