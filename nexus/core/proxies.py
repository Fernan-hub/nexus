from typing import Any

from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy

from nexus.core.interfaces import SignalProxy
from nexus.core.operation_node import OperationNode
from nexus.protocols import AnnotatedItem


class NeoSignalProxy(SignalProxy):
    def __init__(
        self, neo_proxy: BaseProxy | None = None, data_object: DataObject | None = None
    ) -> None:
        if neo_proxy is None and data_object is None:
            # TODO: Raise a more specific exception here
            raise ValueError("Either neo_proxy or data_object must be provided.")

        annotated_item: AnnotatedItem = (
            neo_proxy if neo_proxy is not None else data_object
        )
        super().__init__(annotated_item.annotations)
        self._native_proxy: BaseProxy | None = neo_proxy
        self._data_object: DataObject | None = data_object

    def load(self) -> DataObject:
        if self._cache is None:
            self._cache: DataObject = (
                self._native_proxy.load()
                if self._native_proxy is not None
                else self._data_object
            )
            if "name" in self._annotations:
                self._cache.name = self._annotations["name"]
        return self._cache


class ComputedSignalProxy(SignalProxy):
    def __init__(
        self,
        operation_node: OperationNode,
        output_index: int,
        annotations: dict[str, Any],
    ) -> None:
        super().__init__(annotations=annotations)
        self._operation_node = operation_node
        self._output_index = output_index

    def load(self) -> DataObject:
        if self._cache is None:
            output_signals = self._operation_node.compute()
            self._cache = output_signals[self._output_index]

        return self._cache
