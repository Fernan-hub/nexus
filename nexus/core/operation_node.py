from neo.core.dataobject import DataObject

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.core.interfaces import SignalProxy
from nexus.processing.interfaces import ProcessingStrategy


class OperationNode:
    def __init__(
        self,
        parent_proxies: dict[str, list[SignalProxy]],
        processing_strategy: ProcessingStrategy,
        annotation_strategy: AnnotationStrategy,
    ) -> None:
        self._parent_proxies = parent_proxies
        self._processing_strategy = processing_strategy
        self._annotation_strategy = annotation_strategy
        self._signals_cache: list[DataObject] | None = None

    def compute(self) -> list[DataObject]:
        if self._signals_cache is not None:
            return self._signals_cache

        parent_signals = {
            key: [proxy.load() for proxy in proxies]
            for key, proxies in self._parent_proxies.items()
        }
        input_data = self._processing_strategy.data_input_type(**parent_signals)
        processed_signals = self._processing_strategy.apply(input_data)
        self._annotation_strategy.annotate(processed_signals)
        self._signals_cache = processed_signals
        return self._signals_cache
