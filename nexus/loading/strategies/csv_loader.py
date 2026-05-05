from dataclasses import dataclass

from neo.core.dataobject import DataObject
from neo.io.proxyobjects import BaseProxy
from neo.io import AsciiSignalIO

from nexus.loading.interfaces import DataLoader
from nexus.loading.models import DataLoaderConfig


@dataclass
class CSVLoaderConfig(DataLoaderConfig):
    delimiter: str = ","
    use_cols: list[int] | None = None
    skip_rows: int = 0
    time_column: int | None = None


class CSVLoader(DataLoader):
    def __init__(self, config: CSVLoaderConfig) -> None:
        super().__init__(config)

    def load_data(self) -> list[BaseProxy] | list[DataObject]:
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
