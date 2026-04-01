from typing import Any

SingleCriteria = dict[str, Any]
Criteria = SingleCriteria | list[SingleCriteria]
