"""Estimator configuration classes for transfer entropy estimation."""

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
    """Abstract base configuration for transfer entropy estimators.

    Parameters
    ----------
    base : LogBaseType, optional
        Logarithm base for entropy computation, by default 2.
    step_size : int, optional
        Step size between consecutive samples, by default 1.
    src_hist_len : int, optional
        History length for the source signal, by default 1.
    dest_hist_len : int, optional
        History length for the destination signal, by default 1.
    prop_time : int, optional
        Propagation delay (lag) from source to destination, by default 0.
    offset : int, optional
        Time offset applied before estimation, by default 0.
    """

    step_size: int = 1
    src_hist_len: int = 1
    dest_hist_len: int = 1
    prop_time: int = 0
    offset: int = 0


@dataclass
class DiscreteTransferEntropyConfig(TransferEntropyConfig):
    """Estimator configuration for discrete transfer entropy estimation.

    Parameters
    ----------
    base : LogBaseType, optional
        Logarithm base for entropy computation, by default 2.
    step_size : int, optional
        Step size between consecutive samples, by default 1.
    src_hist_len : int, optional
        History length for the source signal, by default 1.
    dest_hist_len : int, optional
        History length for the destination signal, by default 1.
    prop_time : int, optional
        Propagation delay (lag) from source to destination, by default 0.
    offset : int, optional
        Time offset applied before estimation, by default 0.
    """

    def get_approach_name(self) -> str:
        """Return the approach identifier for discrete TE.

        Returns
        -------
        str
            The string ``'discrete'``.
        """
        return "discrete"

    def get_estimator_class(self) -> TransferEntropyEstimator:
        """Return the discrete transfer entropy estimator class.

        Returns
        -------
        TransferEntropyEstimator
            ``DiscreteTEEstimator``.
        """
        return DiscreteTEEstimator

    def get_conditional_estimator_class(self) -> TransferEntropyEstimator:
        """Return the discrete conditional transfer entropy estimator class.

        Returns
        -------
        TransferEntropyEstimator
            ``DiscreteCTEEstimator``.
        """
        return DiscreteCTEEstimator


@dataclass
class KernelTransferEntropyConfig(TransferEntropyConfig, KernelEstimatorConfigMixin):
    """Estimator configuration for kernel density transfer entropy estimation.

    Parameters
    ----------
    base : LogBaseType, optional
        Logarithm base for entropy computation, by default 2.
    step_size : int, optional
        Step size between consecutive samples, by default 1.
    src_hist_len : int, optional
        History length for the source signal, by default 1.
    dest_hist_len : int, optional
        History length for the destination signal, by default 1.
    prop_time : int, optional
        Propagation delay (lag) from source to destination, by default 0.
    offset : int, optional
        Time offset applied before estimation, by default 0.
    bandwidth : float or int, optional
        Bandwidth parameter for the kernel estimator, by default 1.0.
    kernel : KernelType, optional
        Kernel function type, by default KernelType.GAUSSIAN.
    workers : int, optional
        Number of parallel workers for estimation, by default 1.
    """

    def get_approach_name(self) -> str:
        """Return the approach identifier for kernel TE.

        Returns
        -------
        str
            The string ``'kernel'``.
        """
        return "kernel"

    def get_estimator_class(self) -> TransferEntropyEstimator:
        """Return the kernel transfer entropy estimator class.

        Returns
        -------
        TransferEntropyEstimator
            ``KernelTEEstimator``.
        """
        return KernelTEEstimator

    def get_conditional_estimator_class(self) -> TransferEntropyEstimator:
        """Return the kernel conditional transfer entropy estimator class.

        Returns
        -------
        TransferEntropyEstimator
            ``KernelCTEEstimator``.
        """
        return KernelCTEEstimator

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
