from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy

from nexus.core.interfaces import SignalProxy
from nexus.core.operation_node import OperationNode


class NeoSignalProxy(SignalProxy):
    def __init__(
        self, neo_proxy: BaseProxy | None = None, data_object: DataObject | None = None
    ) -> None:
        super().__init__()
        if neo_proxy is None and data_object is None:
            # TODO: Raise a more specific exception here
            raise ValueError("Either neo_proxy or data_object must be provided.")
        self._native_proxy: BaseProxy | None = neo_proxy
        self._data_object: DataObject | None = data_object

    def load(self) -> DataObject:
        if self._cache is None:
            self._cache: DataObject = (
                self._native_proxy.load()
                if self._native_proxy is not None
                else self._data_object
            )
        return self._cache


class ComputedSignalProxy(SignalProxy):
    def __init__(self, operation_node: OperationNode) -> None:
        super().__init__()
        self._operation_node = operation_node

    def _matches_annotations(self, signal: DataObject) -> bool:
        return self.annotations == signal.annotations

    def load(self) -> DataObject:
        if self._cache is None:
            sibling_signals = self._operation_node.compute()
            for signal in sibling_signals:
                if self._matches_annotations(signal):
                    self._cache = signal
                    break
            else:
                # TODO: Raise a more specific exception here
                raise ValueError(
                    "No computed signal matches the annotations of the proxy."
                )

        return self._cache
