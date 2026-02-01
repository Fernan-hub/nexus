from typing import Any, Sequence
import neo

from interfaces.data_loader import DataLoader
from nexus.interfaces.annotation_strategy import AnnotationStrategy
from nexus.models.criteria import Criteria


class NeuroData:
    def __init__(self, block: neo.Block | None = None) -> None:
        self.__block = block if block is not None else neo.Block()

    def get_all_signals(self) -> list[Any]:
        return []

    def get_signals_by_criteria(self, criteria: Criteria) -> list[Any]:
        return []

    def remove_signals(self, signals: Sequence[Any]) -> None:
        pass

    def add_signals(self, signals: Sequence[Any]) -> None:
        pass

    def load_from_file(self, loader: DataLoader, annotator: AnnotationStrategy) -> None:
        seg = loader.load()
        annotator.annotate(seg)
        self.__block.segments.append(seg)
