from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.exportation.interfaces import ExportationStrategy


class ResultsExporter:
    def __init__(self, analysis_results: LazyAnalysisResult):
        self._lazy_analysis_results = analysis_results

    def export_result(self, exporter_strategy: ExportationStrategy) -> None:
        self._lazy_analysis_results.accept(exporter_strategy)
