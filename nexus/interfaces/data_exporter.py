from abc import ABC, abstractmethod
from nexus.models.analysis_result import AnalysisResult


class DataExporter(ABC):
	@abstractmethod
	def export(self, results: AnalysisResults) -> None:
		pass