from nexus.annotation.interfaces import AnnotationStrategy
from nexus.core.neuro_data import NeuroData
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


class Processor:
    def __init__(self, data: NeuroData) -> None:
        self._data = data

    def process(
        self,
        processing_strategy: ProcessingStrategy,
        annotation_strategy: AnnotationStrategy,
        criteria: Criteria | None = None,
    ) -> None:
        proxies = self._data.get_proxies_by_criteria(criteria)
        computed_proxies = processing_strategy.defer_application(
            proxies, annotation_strategy
        )
        annotation_strategy.annotate(computed_proxies)
        self._data.register_proxies(computed_proxies)
