"""Estimator configuration classes for mutual information estimation."""

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
    """Abstract base configuration for mutual information estimators.

    Parameters
    ----------
    base : LogBaseType, optional
        Logarithm base for entropy computation, by default 2.
    offset : int, optional
        Time offset between the two signals, by default 0.
    normalize : bool, optional
        Whether to normalize MI by the joint entropy, by default False.
    """

    offset: int = 0
    normalize: bool = False


@dataclass
class DiscreteMutualInformationConfig(MutualInformationConfig):
    """Estimator configuration for discrete mutual information estimation.

    Parameters
    ----------
    base : LogBaseType, optional
        Logarithm base for entropy computation, by default 2.
    offset : int, optional
        Time offset between the two signals, by default 0.
    """

    def get_approach_name(self) -> str:
        """Return the approach identifier for discrete MI.

        Returns
        -------
        str
            The string ``'discrete'``.
        """
        return "discrete"

    def get_estimator_class(self) -> MutualInformationEstimator:
        """Return the discrete mutual information estimator class.

        Returns
        -------
        MutualInformationEstimator
            ``DiscreteMIEstimator``.
        """
        return DiscreteMIEstimator

    def get_conditional_estimator_class(self) -> MutualInformationEstimator:
        """Return the discrete conditional mutual information estimator class.

        Returns
        -------
        MutualInformationEstimator
            ``DiscreteCMIEstimator``.
        """
        return DiscreteCMIEstimator

    def get_estimator_kwargs(self) -> dict[str, Any]:
        """Return kwargs, omitting normalize (hardcoded by the discrete estimator).

        Returns
        -------
        dict[str, Any]
            Keyword arguments for the estimator constructor.
        """
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
    """Estimator configuration for kernel density mutual information estimation.

    Parameters
    ----------
    base : LogBaseType, optional
        Logarithm base for entropy computation, by default 2.
    offset : int, optional
        Time offset between the two signals, by default 0.
    normalize : bool, optional
        Whether to normalize MI by the joint entropy, by default False.
    bandwidth : float or int, optional
        Bandwidth parameter for the kernel estimator, by default 1.0.
    kernel : KernelType, optional
        Kernel function type, by default KernelType.GAUSSIAN.
    workers : int, optional
        Number of parallel workers for estimation, by default 1.
    """

    def get_approach_name(self) -> str:
        """Return the approach identifier for kernel MI.

        Returns
        -------
        str
            The string ``'kernel'``.
        """
        return "kernel"

    def get_estimator_class(self) -> MutualInformationEstimator:
        """Return the kernel mutual information estimator class.

        Returns
        -------
        MutualInformationEstimator
            ``KernelMIEstimator``.
        """
        return KernelMIEstimator

    def get_conditional_estimator_class(self) -> MutualInformationEstimator:
        """Return the kernel conditional mutual information estimator class.

        Returns
        -------
        MutualInformationEstimator
            ``KernelCMIEstimator``.
        """
        return KernelCMIEstimator

    def get_estimator_kwargs(self) -> dict[str, Any]:
        """Return estimator kwargs with the kernel enum resolved to its string value.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for the estimator constructor.
        """
        kwargs = super().get_estimator_kwargs()
        kwargs["kernel"] = self.kernel.value
        return kwargs
