import uuid

from abc import ABC, abstractmethod
from neo.core.dataobject import DataObject

from nexus.types import Annotation


class SignalProxy(ABC):
    def __init__(self, annotations: Annotation | None = None) -> None:
        self._id = uuid.uuid4().hex
        self._annotations: Annotation = annotations if annotations is not None else {}
        self._cache: DataObject | None = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def annotations(self) -> Annotation:
        return self._annotations

    @abstractmethod
    def load(self) -> DataObject:
        pass
