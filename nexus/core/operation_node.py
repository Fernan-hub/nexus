from neo.core.dataobject import DataObject

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.core.interfaces.processing_strategy import ProcessingStrategy
from nexus.core.interfaces.signal_proxy import SignalProxy


class OperationNode:
    def __init__(
        self,
        parent_proxies: list[SignalProxy],
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

        parent_signals = [proxy.load() for proxy in self._parent_proxies]
        processed_signals = self._processing_strategy.apply(parent_signals)
        self._annotation_strategy.annotate(processed_signals)
        self._signals_cache = processed_signals
        return self._signals_cache
