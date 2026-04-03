from abc import ABC, abstractmethod
from neo.io.proxyobjects import BaseProxy


class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> list[BaseProxy]:
        pass
