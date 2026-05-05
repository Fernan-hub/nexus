from dataclasses import dataclass, field
from quantities.unitquantity import UnitQuantity, UnitTime

import quantities as pq


@dataclass
class DataLoaderConfig:
    file_path: str
    sampling_rate: UnitQuantity
    units: UnitQuantity
    t_start: UnitTime = field(default_factory=lambda: 0.0 * pq.s)
    time_units: UnitTime = field(default_factory=lambda: pq.s)
