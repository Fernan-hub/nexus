from neo.core import Block
from neo.core.container import filterdata
from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.core.interfaces import SignalProxy
from nexus.core.proxies import NeoSignalProxy
from nexus.loading.interfaces import DataLoader
from nexus.protocols import AnnotatedItem
from nexus.types import Criteria


class NeuroData:
    def __init__(self) -> None:
        self._proxy_registry: dict[str, SignalProxy] = {}

    def get_proxies_by_criteria(self, criteria: Criteria | None) -> list[SignalProxy]:
        return filterdata(self._proxy_registry.values(), criteria)

    def get_proxies_by_criteria_dict(
        self, criteria_dict: dict[str, Criteria]
    ) -> dict[str, list[SignalProxy]]:
        return {
            key: self.get_proxies_by_criteria(criterion)
            for key, criterion in criteria_dict.items()
        }

    def get_proxy_by_id(self, proxy_id: str) -> SignalProxy | None:
        return self._proxy_registry.get(proxy_id)

    def register_proxy(self, proxy: SignalProxy) -> None:
        self._proxy_registry[proxy.id] = proxy

    def register_proxies(self, proxies: list[SignalProxy]) -> None:
        for proxy in proxies:
            self.register_proxy(proxy)

    def load_from_file(
        self, data_loader: DataLoader, annotation_strategy: AnnotationStrategy
    ) -> None:
        loaded_data = data_loader.load_data()
        signal_proxies: list[AnnotatedItem] = []
        for data in loaded_data:
            if isinstance(data, BaseProxy):
                signal_proxy = NeoSignalProxy(neo_proxy=data)
            elif isinstance(data, DataObject):
                signal_proxy = NeoSignalProxy(data_object=data)
            else:
                continue
            self.register_proxy(signal_proxy)
            signal_proxies.append(signal_proxy)
        annotation_strategy.annotate(signal_proxies)

    def to_neo_block(self) -> Block:
        # TODO: Implement conversion to Neo block
        return Block()
