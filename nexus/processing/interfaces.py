from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from neo.core.dataobject import DataObject

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.core.interfaces import SignalProxy
from nexus.common.interfaces import FilterCriteria

ProxyInputT = TypeVar("ProxyInputT")
DataInputT = TypeVar("DataInputT")


class ProcessingStrategy(ABC, Generic[ProxyInputT, DataInputT]):
    @property
    @abstractmethod
    def filter_criteria_type(self) -> type[FilterCriteria]:
        pass

    @property
    @abstractmethod
    def proxy_input_type(self) -> type[ProxyInputT]:
        pass

    @property
    @abstractmethod
    def data_input_type(self) -> type[DataInputT]:
        pass

    @abstractmethod
    def defer_application(
        self, inputs: ProxyInputT, annotation_strategy: AnnotationStrategy
    ) -> list[SignalProxy]:
        pass

    @abstractmethod
    def apply(self, inputs: DataInputT) -> list[DataObject]:
        pass
