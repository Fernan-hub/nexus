from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from neo.core.dataobject import DataObject

from nexus.common.interfaces import FilterCriteria
from nexus.models import NodeDefinition

ProxyInputT = TypeVar("ProxyInputT")
DataInputT = TypeVar("DataInputT")


class ProcessingStrategy(ABC, Generic[ProxyInputT, DataInputT]):

    supported_data_object_types: list[type[DataObject]] = []

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

    def validate_data_objects(self, data_objects: list[DataObject]) -> None:
        for data_object in data_objects:
            if not any(
                issubclass(type(data_object), supported_type)
                for supported_type in self.supported_data_object_types
            ):
                raise TypeError(
                    f"{type(data_object).__name__} is not supported by {type(self).__name__}"
                )

    @abstractmethod
    def infer_execution_plan(self, input_proxies: ProxyInputT) -> list[NodeDefinition]:
        pass

    @abstractmethod
    def apply(self, input_data: DataInputT) -> list[DataObject]:
        pass
