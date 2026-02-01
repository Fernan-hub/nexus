from abc import ABC, abstractmethod
from typing import Any, Sequence


class ProcessingStrategy(ABC):
	@abstractmethod
	def apply(self, signals: Sequence[Any]) -> None:
		pass