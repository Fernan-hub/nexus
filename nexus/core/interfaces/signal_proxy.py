from abc import abstractmethod
import uuid
from neo.core.dataobject import DataObject

from nexus.core.interfaces.annotated_item import AnnotatedItem


class SignalProxy(AnnotatedItem):
    def __init__(self) -> None:
        super().__init__()
        self._id = uuid.uuid4().hex
        self._cache: DataObject | None = None

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def load(self) -> DataObject:
        pass
