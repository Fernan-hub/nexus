"""Facade that drives export of a LazyAnalysisResult via an ExportationStrategy."""

from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.exportation.interfaces import ExportationStrategy


class ResultsExporter:
    """Drives the exportation phase of the Nexus pipeline.

    Thin coordinator: calls accept() on the LazyAnalysisResult with the given
    strategy, which triggers computation (if not yet done) and then export.

    Parameters
    ----------
    analysis_results : LazyAnalysisResult
        The deferred result to be exported.
    """

    def __init__(self, analysis_results: LazyAnalysisResult):
        self._lazy_analysis_results = analysis_results

    def export_result(self, exporter_strategy: ExportationStrategy) -> None:
        """Trigger computation and export via the given ExportationStrategy.

        Parameters
        ----------
        exporter_strategy : ExportationStrategy
            Visitor that receives and writes or renders the analysis result.
        """
        self._lazy_analysis_results.accept(exporter_strategy)
