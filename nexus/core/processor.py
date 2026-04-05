from nexus.annotation.interfaces import AnnotationStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.core.neuro_data import NeuroData
from nexus.processing.interfaces import ProcessingStrategy


class Processor:
    def __init__(self, data: NeuroData) -> None:
        self._data = data

    def process(
        self,
        processing_strategy: ProcessingStrategy,
        annotation_strategy: AnnotationStrategy,
        filter_criteria: FilterCriteria,
    ) -> None:
        proxies_by_criteria = self._data.get_proxies_by_criteria_dict(
            filter_criteria.to_dict()
        )
        input_proxies = processing_strategy.proxy_input_type(**proxies_by_criteria)
        computed_proxies = processing_strategy.defer_application(
            input_proxies, annotation_strategy
        )
        annotation_strategy.annotate(computed_proxies)
        self._data.register_proxies(computed_proxies)
