from dataclasses import dataclass


@dataclass
class ExportationStrategyConfig:
    output_file_path: str | None = None
