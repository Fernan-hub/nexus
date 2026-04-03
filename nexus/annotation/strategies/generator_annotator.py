from typing import Generator

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.protocols import AnnotatedItem
from nexus.types import Annotation


class GeneratorAnnotator(AnnotationStrategy):
    def __init__(
        self,
        ann_gen: Generator[Annotation, None, None],
        default_annotation: Annotation | None = None,
    ) -> None:
        self._ann_gen = ann_gen
        self._default_annotation = default_annotation

    def _next_annotation(self) -> Annotation:
        try:
            return next(self._ann_gen)
        except StopIteration as exc:
            if self._default_annotation is not None:
                return self._default_annotation

            # TODO: Raise a more specific exception here
            raise RuntimeError(
                "No more annotations available and no default annotation set."
            ) from exc

    def annotate(self, items: list[AnnotatedItem]) -> None:
        for item in items:
            annotation = self._next_annotation()
            item.annotations.update(annotation)
