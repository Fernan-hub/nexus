"""Abstract base class for all data loaders in the loading phase."""

from abc import ABC, abstractmethod
from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy

from nexus.loading.models import DataLoaderConfig


class DataLoader(ABC):
    """Base class for strategies that read raw data from a file into Neo objects."""

    def __init__(
        self,
        config: DataLoaderConfig,
    ) -> None:
        self._config = config

    @abstractmethod
    def load_data(self) -> list[BaseProxy] | list[DataObject]:
        """Read data from the configured file and return a flat list of Neo objects.

        Returns
        -------
        list[BaseProxy] | list[DataObject]
            All signals found in the file, in file order.
        """
        pass
