"""DataLoader strategy for reading analog signals from ASCII/CSV files via Neo."""

from dataclasses import dataclass, field


import quantities as pq
from quantities.unitquantity import UnitQuantity, UnitTime
from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy
from neo.io import AsciiSignalIO

from nexus.loading.interfaces import DataLoader
from nexus.loading.models import DataLoaderConfig


@dataclass
class CSVLoaderConfig(DataLoaderConfig):
    """Configuration for CSVLoader; maps directly to AsciiSignalIO arguments."""

    sampling_rate: UnitQuantity
    units: UnitQuantity
    t_start: UnitTime = field(default_factory=lambda: 0.0 * pq.s)
    time_units: UnitTime = field(default_factory=lambda: pq.s)
    delimiter: str = ","
    use_cols: list[int] | None = None
    skip_rows: int = 0
    time_column: int | None = None


class CSVLoader(DataLoader):
    """Loads analog signals from a delimited ASCII file using Neo's AsciiSignalIO."""

    def __init__(self, config: CSVLoaderConfig) -> None:
        super().__init__(config)

    def load_data(self) -> list[BaseProxy] | list[DataObject]:
        """Read all analog signals from the CSV file and return them as a flat list.

        Returns
        -------
        list[BaseProxy] | list[DataObject]
            All analog and irregularly sampled signals found in the file.
        """
        reader = AsciiSignalIO(
            filename=self._config.file_path,
            delimiter=self._config.delimiter,
            usecols=self._config.use_cols,
            skiprows=self._config.skip_rows,
            timecolumn=self._config.time_column,
            sampling_rate=self._config.sampling_rate,
            t_start=self._config.t_start,
            units=self._config.units,
            time_units=self._config.time_units,
        )
        segment = reader.read_segment()
        return segment.analogsignals + segment.irregularlysampledsignals
