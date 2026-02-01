from typing import Any
import neo

from interfaces.data_loader import DataLoader
from nexus.models.criteria import Criteria
from nexus.models.signal_metadata import SignalMetadata


class NeuroData:
	def __init__(self, block: neo.Block | None = None) -> None:
		self.__block = block if block is not None else neo.Block()
  
	def get_all_signals(self) -> list[Any]:
		return []
  
	def get_signals_by_criteria(self, criteria: Criteria) -> list[Any]:
		return []

	def remove_signals(self, signals: list[Any]) -> None:
		pass

	def add_signals(self, signals: list[Any]) -> None:
		pass
	
	def load_from_file(
		self,
		loader: DataLoader,
		metadata: SignalMetadata | None = None
	) -> None:
		seg = loader.load()
		if metadata is not None:
			self.__annotate(seg, metadata)
		self.__block.segments.append(seg)
	
	@staticmethod
	def __annotate(seg: neo.Segment, metadata: SignalMetadata) -> None:
		pass