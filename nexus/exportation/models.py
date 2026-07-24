"""Configuration dataclasses for exportation strategies."""

from dataclasses import dataclass


@dataclass
class ExportationStrategyConfig:
    """Base configuration shared by all exportation strategies.

    Parameters
    ----------
    output_file_path : str or None, optional
        Destination path for the exported file. When None, the strategy falls
        back to using the analysis algorithm name as the file path.
    """

    output_file_path: str | None = None
