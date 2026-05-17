from neo.core import Block, Segment
from neo.core.analogsignal import AnalogSignal
from neo.core.container import filterdata
from neo.core.dataobject import DataObject
from neo.core.spiketrain import SpikeTrain
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

    def to_neo_block(self, criteria_dict: dict[str, Criteria] | None = None) -> Block:
        block = Block()
        if criteria_dict is None:
            proxies = list(self._proxy_registry.values())
            self._load_proxies_into_new_segment(block, proxies)
        else:
            proxies_by_criteria = self.get_proxies_by_criteria_dict(criteria_dict)
            for name, proxies in proxies_by_criteria.items():
                self._load_proxies_into_new_segment(block, proxies, seg_name=name)
        return block

    def _load_proxies_into_new_segment(
        self, block: Block, proxies: list[SignalProxy], seg_name: str | None = None
    ) -> None:
        seg = Segment(name=seg_name)
        block.segments.append(seg)
        seg.block = block
        for proxy in proxies:
            data = proxy.load()
            self._load_into_segment(data, seg)

    @staticmethod
    def _load_into_segment(data: DataObject, seg: Segment) -> None:
        if isinstance(data, AnalogSignal):
            seg.analogsignals.append(data)
        elif isinstance(data, SpikeTrain):
            seg.spiketrains.append(data)
        else:
            return
        data.segment = seg
