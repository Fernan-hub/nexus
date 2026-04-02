from nexus.core.neuro_data import NeuroData
from nexus.core.interfaces.annotation_strategy import AnnotationStrategy
from nexus.core.interfaces.processing_strategy import ProcessingStrategy
from nexus.models.criteria import Criteria


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
