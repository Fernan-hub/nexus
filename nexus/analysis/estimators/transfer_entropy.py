from abc import ABC
from dataclasses import dataclass
from typing import Any

from infomeasure.estimators.base import TransferEntropyEstimator
from infomeasure.estimators.transfer_entropy import (
    DiscreteTEEstimator,
    DiscreteCTEEstimator,
    KernelTEEstimator,
    KernelCTEEstimator,
)
from nexus.analysis.estimators.mixins import KernelEstimatorConfigMixin
from nexus.analysis.interfaces import EstimatorConfigBase


@dataclass
class TransferEntropyConfig(EstimatorConfigBase, ABC):
    step_size: int = 1
    src_hist_len: int = 1
    dest_hist_len: int = 1
    prop_time: int = 0
    offset: int = 0


@dataclass
class DiscreteTransferEntropyConfig(TransferEntropyConfig):
    def get_approach_name(self) -> str:
        return "discrete"

    def get_estimator_class(self) -> TransferEntropyEstimator:
        return DiscreteTEEstimator

    def get_conditional_estimator_class(self) -> TransferEntropyEstimator:
        return DiscreteCTEEstimator


@dataclass
class KernelTransferEntropyConfig(TransferEntropyConfig, KernelEstimatorConfigMixin):
    def get_approach_name(self) -> str:
        return "kernel"

    def get_estimator_class(self) -> TransferEntropyEstimator:
        return KernelTEEstimator

    def get_conditional_estimator_class(self) -> TransferEntropyEstimator:
        return KernelCTEEstimator

    def get_estimator_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_estimator_kwargs()
        kwargs["kernel"] = self.kernel.value
        return kwargs
