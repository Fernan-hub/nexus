"""Analysis result holding a DataFrame of per-signal values."""

from pandas import DataFrame

from nexus.analysis.interfaces import AnalysisResult
from nexus.exportation.interfaces import ExportationStrategy


class VectorResult(AnalysisResult):
    """Analysis result that wraps a DataFrame of per-signal scalar values.

    Parameters
    ----------
    algorithm : str
        Name of the analysis algorithm that produced this result.
    vector : DataFrame
        DataFrame where each row corresponds to one input signal.
    """

    def __init__(self, algorithm: str, vector: DataFrame) -> None:
        super().__init__(algorithm)
        self.vector = vector

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        """Dispatch this result to the exporter's vector result handler.

        Parameters
        ----------
        exporter_strategy : ExportationStrategy
            The exporter that will consume this result.
        """
        exporter_strategy.export_vector_result(self)
