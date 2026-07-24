"""Concrete SignalProxy implementations for Neo-backed and computed signals."""

from typing import Any

from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy

from nexus.core.interfaces import SignalProxy
from nexus.core.operation_node import OperationNode
from nexus.protocols import AnnotatedItem


class NeoSignalProxy(SignalProxy):
    """SignalProxy backed by a Neo BaseProxy or a pre-loaded DataObject.

    Accepts either a lazy Neo I/O proxy (loaded on demand) or a DataObject
    that is already in memory. Exactly one of the two must be supplied.

    Parameters
    ----------
    neo_proxy : BaseProxy or None, optional
        Lazy Neo I/O proxy whose load() materialises the signal.
    data_object : DataObject or None, optional
        Pre-loaded Neo data object used directly without further I/O.
    """

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
        """Load and cache the DataObject, propagating the proxy's name annotation.

        Returns
        -------
        DataObject
            The materialised Neo data object, loaded from the Neo proxy or taken
            directly from the pre-loaded data object.
        """
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
    """SignalProxy whose data is produced by an OperationNode on first access.

    Points to a specific output index of a shared OperationNode so that
    multiple output signals from the same strategy call share one computation.

    Parameters
    ----------
    operation_node : OperationNode
        The shared computation node that produces this signal's data.
    output_index : int
        Index into the list returned by OperationNode.compute() for this proxy.
    annotations : dict of {str: Any}
        Metadata attached to this proxy, typically inherited from the strategy.
    """

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
        """Trigger the shared OperationNode and return this proxy's output slice.

        Returns
        -------
        DataObject
            The output signal at output_index from the shared OperationNode.
        """
        if self._cache is None:
            output_signals = self._operation_node.compute()
            self._cache = output_signals[self._output_index]

        return self._cache
