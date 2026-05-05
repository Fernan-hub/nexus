from dataclasses import dataclass

from nexus.analysis.constants import KernelType


@dataclass
class KernelEstimatorConfigMixin:
    bandwidth: float | int = 1.0
    kernel: KernelType = KernelType.GAUSSIAN
    workers: int = 1
