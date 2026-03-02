from __future__ import annotations

"""Base abstractions for all metric implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class MetricResult:
    """Container for scalar metric outputs and an optional chart object."""

    values: dict[str, float | str]
    chart: object | None = None


class Metric(ABC):
    """Abstract metric contract implemented by each metric module."""

    metric_id: str
    label: str

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> MetricResult:
        """Compute scalar outputs and optional chart object."""
