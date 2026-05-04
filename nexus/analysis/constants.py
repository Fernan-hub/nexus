from enum import Enum


class KernelType(str, Enum):
    GAUSSIAN = "gaussian"
    BOX = "box"
