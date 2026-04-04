from typing import Any, TypeGuard
from neo.core.filters import FilterCondition


Annotation = dict[str, Any]

SingleCriteria = dict[str, FilterCondition]
Criteria = SingleCriteria | list[SingleCriteria]


def is_single_criteria(val: object) -> TypeGuard[SingleCriteria]:
    return isinstance(val, dict) and all(
        isinstance(v, FilterCondition) for v in val.values()
    )


def is_criteria(val: object) -> TypeGuard[Criteria]:
    if is_single_criteria(val):
        return True
    if isinstance(val, list) and all(is_single_criteria(item) for item in val):
        return True
    return False
