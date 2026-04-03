from typing import Any


Annotation = dict[str, Any]

SingleCriteria = dict[str, Any]
Criteria = SingleCriteria | list[SingleCriteria]
