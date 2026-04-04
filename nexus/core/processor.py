from nexus.annotation.interfaces import AnnotationStrategy
from nexus.core.neuro_data import NeuroData
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria, is_criteria


class Processor:
    def __init__(self, data: NeuroData) -> None:
        self._data = data

    def process(
        self,
        processing_strategy: ProcessingStrategy,
        annotation_strategy: AnnotationStrategy,
        criteria: dict[str, Criteria] | Criteria | None = None,
    ) -> None:
        if isinstance(criteria, dict) and not is_criteria(criteria):
            proxies_by_criteria = self._data.get_proxies_by_criteria_dict(criteria)
            computed_proxies = processing_strategy.defer_application(
                annotation_strategy, **proxies_by_criteria
            )
        else:
            proxies = self._data.get_proxies_by_criteria(criteria)
            computed_proxies = processing_strategy.defer_application(
                annotation_strategy, proxies
            )
        annotation_strategy.annotate(computed_proxies)
        self._data.register_proxies(computed_proxies)
