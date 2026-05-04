from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.exporting.interfaces import ExportationStrategy


class ResultsExporter:
    def __init__(self, analysis_results: LazyAnalysisResult):
        self._lazy_analysisresults = analysis_results

    def export_result(self, exporter_strategy: ExportationStrategy) -> None:
        self._lazy_analysisresults.accept(exporter_strategy)
