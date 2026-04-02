from nexus.core.interfaces.exporter_strategy import ExporterStrategy
from nexus.core.lazy_analysis_result import LazyAnalysisResult


class ResultsExporter:
    def __init__(self, analysis_results: LazyAnalysisResult):
        self._lazy_analysisresults = analysis_results

    def export_result(self, exporter_strategy: ExporterStrategy) -> None:
        self._lazy_analysisresults.accept(exporter_strategy)
