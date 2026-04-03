from abc import ABC


class ExporterStrategy(ABC):
    def __init__(self, output_path: str | None) -> None:
        self.output_path = output_path
