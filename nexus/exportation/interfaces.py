"""Abstract base for exportation strategies in the Nexus pipeline."""

from abc import ABC

from typing import TYPE_CHECKING

from nexus.exportation.models import ExportationStrategyConfig

if TYPE_CHECKING:
    from nexus.analysis.interfaces import AnalysisResult
    from nexus.analysis.results import MatrixResult, ScalarResult, VectorResult


class ExportationStrategy(ABC):
    """Visitor that writes or renders an AnalysisResult to an output destination.

    Implements the visitor pattern: each export_*_result method corresponds to
    one concrete AnalysisResult subtype. Subclasses override only the methods
    they support; the default implementations raise NotImplementedError.

    Parameters
    ----------
    config : ExportationStrategyConfig
        Configuration for the exporter, including the optional output file path.
    """

    def __init__(self, config: ExportationStrategyConfig) -> None:
        self._config = config

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        """Export a MatrixResult.

        Parameters
        ----------
        matrix_result : MatrixResult
            The pairwise analysis result to export.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support exporting matrix results."
        )

    def export_vector_result(self, vector_result: "VectorResult") -> None:
        """Export a VectorResult.

        Parameters
        ----------
        vector_result : VectorResult
            The per-signal analysis result to export.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support exporting vector results."
        )

    def export_scalar_result(self, scalar_result: "ScalarResult") -> None:
        """Export a ScalarResult.

        Parameters
        ----------
        scalar_result : ScalarResult
            The single-value analysis result to export.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support exporting scalar results."
        )

    def _get_output_file_path(self, analysis_result: "AnalysisResult") -> str:
        return self._config.output_file_path or analysis_result.algorithm
