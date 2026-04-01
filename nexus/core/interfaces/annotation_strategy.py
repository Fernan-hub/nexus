from abc import ABC, abstractmethod

from nexus.core.interfaces.annotated_item import AnnotatedItem


class AnnotationStrategy(ABC):
    @abstractmethod
    def annotate(self, items: list[AnnotatedItem]) -> None:
        pass
