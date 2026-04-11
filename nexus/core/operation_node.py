from neo.core.dataobject import DataObject

from nexus.core.interfaces import SignalProxy
from nexus.processing.interfaces import ProcessingStrategy


class OperationNode:
    def __init__(
        self,
        input_proxies: dict[str, list[SignalProxy]],
        processing_strategy: ProcessingStrategy,
    ) -> None:
        self._input_proxies = input_proxies
        self._processing_strategy = processing_strategy
        self._signals_cache: list[DataObject] | None = None

    def compute(self) -> list[DataObject]:
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
