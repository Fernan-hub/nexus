from nexus.core.neuro_data import NeuroData
from nexus.interfaces.analysis_strategy import AnalysisStrategy
from nexus.models.results import AnalysisResult
from nexus.models.criteria import Criteria


class Analyzer:
	def __init__(self, processed_data: NeuroData) -> None:
		self.__data = processed_data
	
	def analyze_data(
		self, 
		strategy: AnalysisStrategy,
		criteria: Criteria | None = None,
	) -> AnalysisResult:
		if criteria is not None:
			signals = self.__data.get_signals_by_criteria(criteria)
		else:
			signals = self.__data.get_all_signals()
		return strategy.run_analysis(signals)
