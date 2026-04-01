from abc import ABC, abstractmethod
from neo.core.dataobject import DataObject

from nexus.core.interfaces.annotation_strategy import AnnotationStrategy
from nexus.core.interfaces.signal_proxy import SignalProxy


class ProcessingStrategy(ABC):
    @abstractmethod
    def defer_application(
        self, signal_proxies: list[SignalProxy], annotation_strategy: AnnotationStrategy
    ) -> list[SignalProxy]:
        pass

    @abstractmethod
    def apply(self, signals: list[DataObject]) -> list[DataObject]:
        pass
