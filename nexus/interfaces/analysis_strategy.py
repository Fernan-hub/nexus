from abc import ABC, abstractmethod
from typing import Any, Sequence

from nexus.models.results import AnalysisResult


class AnalysisStrategy(ABC):
	@abstractmethod
	def run_analysis(self, signals: Sequence[Any]) -> AnalysisResult:
		pass
