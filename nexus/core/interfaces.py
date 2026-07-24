"""Abstract base for all signal proxies in the Nexus pipeline."""

import uuid

from abc import ABC, abstractmethod
from neo.core.dataobject import DataObject

from nexus.types import Annotation


class SignalProxy(ABC):
    """Lazy, annotated wrapper around a Neo DataObject.

    Every signal in a Nexus pipeline -- whether loaded from disk or derived
    by a processing strategy -- is represented as a SignalProxy. Data is not
    materialised until :meth:`load` is called for the first time.

    Parameters
    ----------
    annotations : Annotation or None, optional
        Initial metadata dict attached to this proxy. Defaults to an empty dict.
    """

    def __init__(self, annotations: Annotation | None = None) -> None:
        self._id = uuid.uuid4().hex
        self._annotations: Annotation = annotations if annotations is not None else {}
        self._cache: DataObject | None = None

    @property
    def id(self) -> str:
        """Unique hex identifier for this proxy, stable across its lifetime.

        Returns
        -------
        str
            Hex string UUID assigned at construction time.
        """
        return self._id

    @property
    def annotations(self) -> Annotation:
        """Metadata dict used as filter keys by NeuroData.get_proxies_by_criteria.

        Returns
        -------
        Annotation
            The annotation dictionary attached to this proxy.
        """
        return self._annotations

    @abstractmethod
    def load(self) -> DataObject:
        """Return the underlying DataObject, loading or computing it if needed.

        Returns
        -------
        DataObject
            The materialised Neo data object for this signal.
        """
        pass
