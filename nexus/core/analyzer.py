from nexus.core.interfaces.analysis_strategy import AnalysisStrategy
from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.core.neuro_data import NeuroData
from nexus.types import Criteria


class Analyzer:
    def __init__(self, data: NeuroData) -> None:
        self._data = data

    def analyze_data(
        self, analysis_strategy: AnalysisStrategy, criteria: Criteria | None = None
    ) -> LazyAnalysisResult:
        input_proxies = self._data.get_proxies_by_criteria(criteria)
        return LazyAnalysisResult(input_proxies, analysis_strategy)
