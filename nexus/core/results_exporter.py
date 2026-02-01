from nexus.interfaces.data_exporter import DataExporter
from nexus.models.analysis_result import AnalysisResult


class ResultsExporter:
	def __init__(self, analysis_result: AnalysisResult) -> None:
		self.__results = analysis_result
	
	def export_results(
		self,
		exporter: DataExporter,
	) -> None:
		exporter.export(self.__results)