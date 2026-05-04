from abc import ABC
from dataclasses import dataclass
from typing import Any

from infomeasure.estimators.base import EntropyEstimator
from infomeasure.estimators.entropy import (
    DiscreteEntropyEstimator,
    KernelEntropyEstimator,
)
from nexus.analysis.estimators.mixins import KernelEstimatorConfigMixin
from nexus.analysis.interfaces import EstimatorConfigBase


@dataclass
class EntropyConfig(EstimatorConfigBase, ABC):
    pass


@dataclass
class DiscreteEntropyConfig(EntropyConfig):
    def get_estimator_class(self) -> EntropyEstimator:
        return DiscreteEntropyEstimator


@dataclass
class KernelEntropyConfig(EntropyConfig, KernelEstimatorConfigMixin):
    def get_estimator_class(self) -> EntropyEstimator:
        return KernelEntropyEstimator

    def get_estimator_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_estimator_kwargs()
        kwargs["kernel"] = self.kernel.value
        return kwargs
