from dataclasses import dataclass

from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy
from neo.io import NWBIO

from nexus.loading.interfaces import DataLoader
from nexus.loading.models import DataLoaderConfig
from nexus.loading.utils import get_all_data_from_blocks


@dataclass
class NWBLoaderConfig(DataLoaderConfig):
    pass


class NWBLoader(DataLoader):
    def __init__(self, config: NWBLoaderConfig) -> None:
        super().__init__(config)

    def load_data(self) -> list[BaseProxy] | list[DataObject]:
        reader = NWBIO(filename=self._config.file_path)
        # lazy=True is broken in neo <=0.14.4: NWBIO.AnalogSignalProxy does not set the
        # .segment attribute that neo.core.objectlist requires when appending to a segment.
        # Switch back to lazy=True once fixed upstream.
        return get_all_data_from_blocks(reader.read_all_blocks(lazy=False))
