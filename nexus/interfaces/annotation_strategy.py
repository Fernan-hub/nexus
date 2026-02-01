from abc import ABC, abstractmethod
from typing import Any, Sequence
import neo

class AnnotationStrategy(ABC):
    @abstractmethod
    def annotate(self, segment: neo.Segment | Sequence[Any]) -> None:
        pass