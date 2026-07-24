"""Lazy computation node shared by all output proxies of one strategy invocation."""

from neo.core.dataobject import DataObject

from nexus.core.interfaces import SignalProxy
from nexus.processing.interfaces import ProcessingStrategy


class OperationNode:
    """Deferred computation that runs a ProcessingStrategy over a fixed set of inputs.

    Multiple ComputedSignalProxy objects may point to the same OperationNode
    (one per output index). The strategy is executed at most once; the result
    list is cached and returned on every subsequent call to :meth:`compute`.

    Parameters
    ----------
    input_proxies : dict of {str: list of SignalProxy}
        Named groups of input proxies matching the strategy's proxy input type.
    processing_strategy : ProcessingStrategy
        The strategy whose apply() method produces the output signals.
    """

    def __init__(
        self,
        input_proxies: dict[str, list[SignalProxy]],
        processing_strategy: ProcessingStrategy,
    ) -> None:
        self._input_proxies = input_proxies
        self._processing_strategy = processing_strategy
        self._signals_cache: list[DataObject] | None = None

    def compute(self) -> list[DataObject]:
        """Run the strategy once, cache the outputs, and return them on every call.

        Returns
        -------
        list of DataObject
            The output signals produced by the processing strategy.
        """
        if self._signals_cache is not None:
            return self._signals_cache

        input_data_dict = {
            key: [proxy.load() for proxy in proxies]
            for key, proxies in self._input_proxies.items()
        }
        input_data = self._processing_strategy.data_input_type(**input_data_dict)
        output_signals = self._processing_strategy.apply(input_data)
        self._signals_cache = output_signals
        return self._signals_cache
