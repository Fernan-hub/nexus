"""Central proxy registry and loading facade for a Nexus pipeline session."""

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
    """Registry that maps proxy IDs to SignalProxy objects for the whole session.

    Acts as the single source of truth for all signals - loaded from files or
    produced by processing strategies. Filtering uses Neo's filterdata mechanism
    so that annotation dicts on each proxy serve as query keys.
    """

    def __init__(self) -> None:
        self._proxy_registry: dict[str, SignalProxy] = {}

    def get_proxies_by_criteria(self, criteria: Criteria | None) -> list[SignalProxy]:
        """Return all proxies whose annotations satisfy criteria via Neo filterdata.

        Parameters
        ----------
        criteria : Criteria or None
            A dict or list of dicts of annotation key-value pairs to match against.
            None returns all registered proxies.

        Returns
        -------
        list of SignalProxy
            Proxies whose annotation dicts match the given criteria.
        """
        return filterdata(self._proxy_registry.values(), criteria)

    def get_proxies_by_criteria_dict(
        self, criteria_dict: dict[str, Criteria]
    ) -> dict[str, list[SignalProxy]]:
        """Apply get_proxies_by_criteria for each named criteria entry in the dict.

        Parameters
        ----------
        criteria_dict : dict of {str: Criteria}
            Mapping of label to criteria; each entry is filtered independently.

        Returns
        -------
        dict of {str: list of SignalProxy}
            Same keys as criteria_dict, values are the matching proxy lists.
        """
        return {
            key: self.get_proxies_by_criteria(criterion)
            for key, criterion in criteria_dict.items()
        }

    def get_proxy_by_id(self, proxy_id: str) -> SignalProxy | None:
        """Return the proxy registered under proxy_id, or None if not found.

        Parameters
        ----------
        proxy_id : str
            The hex UUID string assigned to the proxy at construction time.

        Returns
        -------
        SignalProxy or None
            The matching proxy, or None if proxy_id is not in the registry.
        """
        return self._proxy_registry.get(proxy_id)

    def register_proxy(self, proxy: SignalProxy) -> None:
        """Add a proxy to the registry under its own ID.

        Parameters
        ----------
        proxy : SignalProxy
            The proxy to register; its id property is used as the key.
        """
        self._proxy_registry[proxy.id] = proxy

    def load_from_file(
        self, data_loader: DataLoader, annotation_strategy: AnnotationStrategy
    ) -> None:
        """Load signals from a file, wrap them in proxies, and annotate them.

        Parameters
        ----------
        data_loader : DataLoader
            Strategy that reads raw data from a file and returns Neo objects.
        annotation_strategy : AnnotationStrategy
            Strategy that attaches metadata to the newly created proxies.
        """
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
        """Build a Neo Block from loaded proxies, one Segment per criteria entry.

        Parameters
        ----------
        criteria_dict : dict of {str: Criteria} or None, optional
            When provided, each key becomes a named Segment containing the proxies
            that match the corresponding criteria. When None, all proxies are placed
            in a single unnamed Segment.

        Returns
        -------
        Block
            Neo Block whose segments hold the materialised DataObjects.
        """
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
