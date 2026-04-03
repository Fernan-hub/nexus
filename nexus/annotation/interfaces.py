from abc import ABC, abstractmethod

from nexus.protocols import AnnotatedItem


class AnnotationStrategy(ABC):
    @abstractmethod
    def annotate(self, items: list[AnnotatedItem]) -> None:
        pass
