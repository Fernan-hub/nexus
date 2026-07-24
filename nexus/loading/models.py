"""Configuration dataclasses shared across all loader strategies."""

from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    """Base config for all data loaders; subclasses add format-specific fields."""

    file_path: str
