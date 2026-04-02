from abc import ABC
from typing import Any


class AnnotatedItem(ABC):
    def __init__(self) -> None:
        self._annotations: dict[str, Any] = {}

    @property
    def annotations(self) -> dict[str, Any]:
        return self._annotations

    def add_annotation(self, key: str, value: Any) -> None:
        self._annotations[key] = value

    def update_annotations(self, annotations: dict[str, Any]) -> None:
        self._annotations.update(annotations)
