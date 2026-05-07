from nexus.analysis.interfaces import AnalysisStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.core.lazy_analysis_result import LazyAnalysisResult
from nexus.core.neuro_data import NeuroData


class Analyzer:
    def __init__(self, data: NeuroData) -> None:
        self._data = data

    def analyze_data(
        self, analysis_strategy: AnalysisStrategy, filter_criteria: FilterCriteria
    ) -> LazyAnalysisResult:
        criteria_dict = {
            k: v for k, v in filter_criteria.to_dict().items() if v is not None
        }
        input_proxies = self._data.get_proxies_by_criteria_dict(criteria_dict)
        return LazyAnalysisResult(input_proxies, analysis_strategy)
