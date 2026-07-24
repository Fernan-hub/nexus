"""Abstract base class and type variables for all processing strategies."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from neo.core.dataobject import DataObject

from nexus.common.interfaces import FilterCriteria
from nexus.models import NodeDefinition

ProxyInputT = TypeVar("ProxyInputT")
DataInputT = TypeVar("DataInputT")


class ProcessingStrategy(ABC, Generic[ProxyInputT, DataInputT]):
    """Base class for strategies that derive new signals from existing ones.

    Each concrete strategy defines which input signals it needs, how to group
    them into lazy compute nodes, and the computation to apply when a node is
    eventually resolved at export time.
    """

    supported_data_object_types: list[type[DataObject]] = []

    @property
    @abstractmethod
    def filter_criteria_type(self) -> type[FilterCriteria]:
        """Return the FilterCriteria subclass used to select inputs."""
        pass

    @property
    @abstractmethod
    def proxy_input_type(self) -> type[ProxyInputT]:
        """Return the ProxyInput dataclass that groups selected SignalProxy objects."""
        pass

    @property
    @abstractmethod
    def data_input_type(self) -> type[DataInputT]:
        """Return the DataInput dataclass that groups loaded DataObject instances."""
        pass

    def validate_data_objects(self, data_objects: list[DataObject]) -> None:
        """Raise TypeError if any object's type is not in supported_data_object_types.

        Parameters
        ----------
        data_objects : list[DataObject]
            Objects to validate against supported_data_object_types.
        """
        for data_object in data_objects:
            if not any(
                issubclass(type(data_object), supported_type)
                for supported_type in self.supported_data_object_types
            ):
                raise TypeError(
                    f"{type(data_object).__name__} is not supported by "
                    f"{type(self).__name__}"
                )

    @abstractmethod
    def infer_execution_plan(self, input_proxies: ProxyInputT) -> list[NodeDefinition]:
        """Build the list of lazy compute nodes for the given input proxies.

        Parameters
        ----------
        input_proxies : ProxyInputT
            Proxies selected for this strategy run.

        Returns
        -------
        list[NodeDefinition]
            One NodeDefinition per output signal to be computed.
        """
        pass

    @abstractmethod
    def apply(self, input_data: DataInputT) -> list[DataObject]:
        """Run the strategy computation on one fully loaded execution node.

        Parameters
        ----------
        input_data : DataInputT
            Loaded data objects for one execution node.

        Returns
        -------
        list[DataObject]
            Computed output signals.
        """
        pass
