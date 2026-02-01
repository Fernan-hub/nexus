from abc import ABC, abstractmethod

import neo
from pathlib import Path
from quantities import UnitQuantity, UnitTime


class DataLoader(ABC):
	def __init__(
		self,
		file_path: str | Path,
		sampling_rate: UnitQuantity,
		units: UnitQuantity,
		time_units: UnitTime,
	) -> None:
		self.__file_path = file_path
		self.__sampling_rate = sampling_rate
		self.__units = units
		self.__time_units = time_units

	@abstractmethod
	def load(self) -> neo.Segment:
		pass