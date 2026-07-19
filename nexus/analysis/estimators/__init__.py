"""Estimator configuration classes."""

from nexus.analysis.estimators.entropy import (
    DiscreteEntropyConfig,
    KernelEntropyConfig,
)
from nexus.analysis.estimators.mutual_information import (
    DiscreteMutualInformationConfig,
    KernelMutualInformationConfig,
)
from nexus.analysis.estimators.transfer_entropy import (
    DiscreteTransferEntropyConfig,
    KernelTransferEntropyConfig,
)

__all__ = [
    "DiscreteEntropyConfig",
    "KernelEntropyConfig",
    "DiscreteMutualInformationConfig",
    "KernelMutualInformationConfig",
    "DiscreteTransferEntropyConfig",
    "KernelTransferEntropyConfig",
]
