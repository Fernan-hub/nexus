from nexus.analysis.interfaces import AnalysisStrategy
from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.core.neuro_data import NeuroData
from nexus.types import Criteria, is_criteria


class Analyzer:
    def __init__(self, data: NeuroData) -> None:
        self._data = data

    def analyze_data(
        self, analysis_strategy: AnalysisStrategy, criteria: Criteria | None = None
    ) -> LazyAnalysisResult:
        if isinstance(criteria, dict) and not is_criteria(criteria):
            input_proxies = self._data.get_proxies_by_criteria_dict(criteria)
        else:
            input_proxies = self._data.get_proxies_by_criteria(criteria)
        return LazyAnalysisResult(input_proxies, analysis_strategy)
