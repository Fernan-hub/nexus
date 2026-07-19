"""Kernel type enumeration for kernel-based estimators."""

from enum import Enum


class KernelType(str, Enum):
    """Kernel function types supported by kernel density estimators."""

    GAUSSIAN = "gaussian"
    BOX = "box"
