from dataclasses import dataclass, asdict

from nexus.types import Criteria


@dataclass
class FilterCriteria:
    def to_dict(self) -> dict[str, Criteria]:
        return asdict(self)
