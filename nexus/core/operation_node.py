from neo.core.dataobject import DataObject

from nexus.core.interfaces.annotation_strategy import AnnotationStrategy
from nexus.core.interfaces.processing_strategy import ProcessingStrategy
from nexus.core.interfaces.signal_proxy import SignalProxy


class OperationNode:
    def __init__(
        self,
        parent_proxies: list[SignalProxy],
        processing_strategy: ProcessingStrategy,
        annotation_strategy: AnnotationStrategy,
    ) -> None:
        self.__parent_proxies = parent_proxies
        self.__processing_strategy = processing_strategy
        self.__annotation_strategy = annotation_strategy
        self.__signals_cache: list[DataObject] | None = None

    def compute(self) -> list[DataObject]:
        if self.__signals_cache is not None:
            return self.__signals_cache

        parent_signals = [proxy.load() for proxy in self.__parent_proxies]
        processed_signals = self.__processing_strategy.apply(parent_signals)
        # TODO: Check how we could cast DataObject instances to AnnotatedItem instances here
        self.__annotation_strategy.annotate(processed_signals)
        self.__signals_cache = processed_signals
        return self.__signals_cache
