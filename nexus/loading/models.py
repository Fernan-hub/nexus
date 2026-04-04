from dataclasses import dataclass
from quantities.unitquantity import UnitQuantity, UnitTime

import quantities as pq


@dataclass
class DataLoaderConfig:
    file_path: str
    sampling_rate: UnitQuantity
    t_start: UnitTime = 0.0 * pq.s
    units: UnitQuantity
    time_units: UnitTime = pq.s
