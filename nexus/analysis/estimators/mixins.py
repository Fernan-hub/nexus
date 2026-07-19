"""Mixin for kernel density estimator configuration fields."""

from dataclasses import dataclass

from nexus.analysis.constants import KernelType


@dataclass
class KernelEstimatorConfigMixin:
    """Mixin that adds kernel density estimation parameters to an estimator config.

    Parameters
    ----------
    bandwidth : float or int, optional
        Bandwidth parameter for the kernel estimator, by default 1.0.
    kernel : KernelType, optional
        Kernel function type, by default KernelType.GAUSSIAN.
    workers : int, optional
        Number of parallel workers for estimation, by default 1.
    """

    bandwidth: float | int = 1.0
    kernel: KernelType = KernelType.GAUSSIAN
    workers: int = 1
