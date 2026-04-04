from abc import ABC, abstractmethod
from neo.core.dataobject import DataObject

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.core.interfaces import SignalProxy


class ProcessingStrategy(ABC):
    @abstractmethod
    def defer_application(
        self,
        annotation_strategy: AnnotationStrategy,
        *args: list[SignalProxy],
        **kwargs: list[SignalProxy],
    ) -> list[SignalProxy]:
        pass

    @abstractmethod
    def apply(
        self, *args: list[SignalProxy], **kwargs: list[DataObject]
    ) -> list[DataObject]:
        pass
