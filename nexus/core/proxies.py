from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy

from nexus.core.interfaces.signal_proxy import SignalProxy
from nexus.core.operation_node import OperationNode


class NeoSignalProxy(SignalProxy):
    def __init__(self, neo_proxy: BaseProxy) -> None:
        super().__init__()
        self._native_proxy = neo_proxy

    def load(self) -> DataObject:
        if self._cache is None:
            self._cache: DataObject = self._native_proxy.load()
        return self._cache


class ComputedSignalProxy(SignalProxy):
    def __init__(self, operation_node: OperationNode) -> None:
        self._operation_node = operation_node

    def _matches_annotations(self, signal: DataObject) -> bool:
        # TODO: Implement actual annotation matching logic here
        return True

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
