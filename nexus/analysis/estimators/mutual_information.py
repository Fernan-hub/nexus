from abc import ABC
from dataclasses import dataclass
from typing import Any

from infomeasure.estimators.base import MutualInformationEstimator
from infomeasure.estimators.mutual_information import (
    DiscreteMIEstimator,
    DiscreteCMIEstimator,
    KernelMIEstimator,
    KernelCMIEstimator,
)
from nexus.analysis.estimators.mixins import KernelEstimatorConfigMixin
from nexus.analysis.interfaces import EstimatorConfigBase


@dataclass
class MutualInformationConfig(EstimatorConfigBase, ABC):
    offset: int = 0
    normalize: bool = False


@dataclass
class DiscreteMutualInformationConfig(MutualInformationConfig):
    def get_approach_name(self) -> str:
        return "discrete"

    def get_estimator_class(self) -> MutualInformationEstimator:
        return DiscreteMIEstimator

    def get_conditional_estimator_class(self) -> MutualInformationEstimator:
        return DiscreteCMIEstimator

    def get_estimator_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_estimator_kwargs()
        # DiscreteMIEstimator hardcodes normalize=False when calling super(),
        # so passing it again via **kwargs causes a "multiple values" TypeError.
        # The kernel estimator accepts normalize as a real parameter, so it
        # must stay in KernelMutualInformationConfig's kwargs.
        kwargs.pop("normalize", None)
        return kwargs


@dataclass
class KernelMutualInformationConfig(
    MutualInformationConfig, KernelEstimatorConfigMixin
):
    def get_approach_name(self) -> str:
        return "kernel"

    def get_estimator_class(self) -> MutualInformationEstimator:
        return KernelMIEstimator

    def get_conditional_estimator_class(self) -> MutualInformationEstimator:
        return KernelCMIEstimator

    def get_estimator_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_estimator_kwargs()
        kwargs["kernel"] = self.kernel.value
        return kwargs
