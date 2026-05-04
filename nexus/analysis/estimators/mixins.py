from nexus.analysis.constants import KernelType


class KernelEstimatorConfigMixin:
    bandwidth: float | int
    kernel: KernelType
    workers: int = 1
