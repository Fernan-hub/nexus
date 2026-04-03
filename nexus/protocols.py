from typing import Protocol

from nexus.types import Annotation


class AnnotatedItem(Protocol):
    annotations: Annotation
