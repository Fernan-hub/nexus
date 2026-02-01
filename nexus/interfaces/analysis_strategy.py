from abc import ABC, abstractmethod
from typing import Any, Sequence

from nexus.models.analysis_result import AnalysisResult


class AnalysisStrategy(ABC):
	@abstractmethod
	def run_analysis(self, signals: Sequence[Any]) -> AnalysisResult:
		pass