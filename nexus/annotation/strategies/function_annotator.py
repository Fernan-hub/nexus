from typing import Callable

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.protocols import AnnotatedItem
from nexus.types import Annotation


class FunctionAnnotator(AnnotationStrategy):
    def __init__(
        self,
        ann_func: Callable[[AnnotatedItem], Annotation],
    ) -> None:
        self._ann_func = ann_func

    def _get_annotation_for_item(self, item: AnnotatedItem) -> Annotation:
        try:
            return self._ann_func(item)
        except Exception as exc:
            raise RuntimeError(
                f"Error while generating annotation for item: {item}"
            ) from exc

    def annotate(self, items: list[AnnotatedItem]) -> None:
        for item in items:
            annotation = self._get_annotation_for_item(item)
            item.annotations.update(annotation)
