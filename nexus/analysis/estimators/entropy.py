"""Estimator configuration classes for entropy estimation."""

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
    """Abstract base configuration for entropy estimators."""


@dataclass
class DiscreteEntropyConfig(EntropyConfig):
    """Estimator configuration for discrete entropy estimation.

    Parameters
    ----------
    base : LogBaseType, optional
        Logarithm base for entropy computation, by default 2.
    """

    def get_approach_name(self) -> str:
        """Return the approach identifier for discrete entropy.

        Returns
        -------
        str
            The string ``'discrete'``.
        """
        return "discrete"

    def get_estimator_class(self) -> EntropyEstimator:
        """Return the discrete entropy estimator class.

        Returns
        -------
        EntropyEstimator
            ``DiscreteEntropyEstimator``.
        """
        return DiscreteEntropyEstimator


@dataclass
class KernelEntropyConfig(EntropyConfig, KernelEstimatorConfigMixin):
    """Estimator configuration for kernel density entropy estimation.

    Parameters
    ----------
    base : LogBaseType, optional
        Logarithm base for entropy computation, by default 2.
    bandwidth : float or int, optional
        Bandwidth parameter for the kernel estimator, by default 1.0.
    kernel : KernelType, optional
        Kernel function type, by default KernelType.GAUSSIAN.
    workers : int, optional
        Number of parallel workers for estimation, by default 1.
    """

    def get_approach_name(self) -> str:
        """Return the approach identifier for kernel entropy.

        Returns
        -------
        str
            The string ``'kernel'``.
        """
        return "kernel"

    def get_estimator_class(self) -> EntropyEstimator:
        """Return the kernel entropy estimator class.

        Returns
        -------
        EntropyEstimator
            ``KernelEntropyEstimator``.
        """
        return KernelEntropyEstimator

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
