import uuid

from abc import ABC, abstractmethod
from typing import Any
from neo.core.dataobject import DataObject


class SignalProxy(ABC):
    def __init__(self) -> None:
        self._id = uuid.uuid4().hex
        self._annotations: dict[str, Any] = {}
        self._cache: DataObject | None = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def annotations(self) -> dict[str, Any]:
        return self._annotations

    @abstractmethod
    def load(self) -> DataObject:
        pass
