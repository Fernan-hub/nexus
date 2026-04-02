from neo.core import Block
from neo.core.container import filterdata

from nexus.core.interfaces.annotated_item import AnnotatedItem
from nexus.core.interfaces.annotation_strategy import AnnotationStrategy
from nexus.core.interfaces.data_loader import DataLoader
from nexus.core.interfaces.signal_proxy import SignalProxy
from nexus.core.proxies import NeoSignalProxy
from nexus.models.criteria import Criteria


class NeuroData:
    def __init__(self) -> None:
        self._proxy_registry: dict[str, SignalProxy] = {}

    def get_proxies_by_criteria(self, criteria: Criteria | None) -> list[SignalProxy]:
        return filterdata(self._proxy_registry.values(), criteria)

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
        proxies = data_loader.load_data()
        signal_proxies: list[AnnotatedItem] = []
        for proxy in proxies:
            signal_proxy = NeoSignalProxy(proxy)
            self.register_proxy(signal_proxy)
            signal_proxies.append(signal_proxy)
        annotation_strategy.annotate(signal_proxies)

    def to_neo_block(self) -> Block:
        # TODO: Implement conversion to Neo block
        return Block()
