"""Analysis result holding a single scalar value."""

from nexus.analysis.interfaces import AnalysisResult
from nexus.exportation.interfaces import ExportationStrategy


class ScalarResult(AnalysisResult):
    """Analysis result that wraps a single scalar measurement.

    Parameters
    ----------
    algorithm : str
        Name of the analysis algorithm that produced this result.
    value : float
        The scalar result value.
    label : str
        Human-readable label describing what the scalar represents.
    """

    def __init__(self, algorithm: str, value: float, label: str) -> None:
        super().__init__(algorithm)
        self.value = value
        self.label = label

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        """Dispatch this result to the exporter's scalar result handler.

        Parameters
        ----------
        exporter_strategy : ExportationStrategy
            The exporter that will consume this result.
        """
        exporter_strategy.export_scalar_result(self)
