from abc import ABC, abstractmethod
from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy

from nexus.loading.models import DataLoaderConfig


class DataLoader(ABC):
    def __init__(
        self,
        config: DataLoaderConfig,
    ) -> None:
        self._config = config

    @abstractmethod
    def load_data(self) -> list[BaseProxy] | list[DataObject]:
        pass
