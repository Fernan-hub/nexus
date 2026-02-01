from nexus.core.neuro_data import NeuroData
from nexus.exceptions import MissingMetadataError
from nexus.interfaces.annotation_strategy import AnnotationStrategy
from nexus.interfaces.processing_strategy import ProcessingStrategy
from nexus.models.criteria import Criteria


class PreProcessor:
	def __init__(self, data: NeuroData) -> None:
		self.__data = data

	def process(
		self,
		strategy: ProcessingStrategy,
		criteria: Criteria | None = None,
		annotator: AnnotationStrategy | None = None,
		remove_after_used: bool = False
	) -> None:
		if criteria is not None:
			signals = self.__data.get_signals_by_criteria(criteria)
		else:
			signals = self.__data.get_all_signals()
		
		new_signals = strategy.apply(signals)
		if remove_after_used is True:
			self.__data.remove_signals(signals)
		
		if new_signals is not None:
			if annotator is None:
				raise MissingMetadataError("An AnnotationStrategy must be provided to annotate the new signals.")
			annotator.annotate(new_signals)
			self.__data.add_signals(new_signals)

	def get_preprocessed_data(self) -> NeuroData:
		return self.__data