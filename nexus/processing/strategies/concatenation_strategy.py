"""Processing strategy that concatenates multiple analog signals into one."""

from dataclasses import dataclass, asdict
from typing import Any

from neo.core import AnalogSignal
from neo.core.dataobject import DataObject

from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.models import NodeDefinition
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class ConcatenationStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting input signals to concatenate."""

    inputs: Criteria | None = None


@dataclass
class ConcatenationStrategyProxyInput:
    """Input container holding SignalProxy objects for concatenation."""

    inputs: list[SignalProxy]


@dataclass
class ConcatenationStrategyDataInput:
    """Input container holding loaded DataObject instances for concatenation."""

    inputs: list[DataObject]


class ConcatenationStrategy(
    ProcessingStrategy[
        ConcatenationStrategyProxyInput,
        ConcatenationStrategyDataInput,
    ]
):
    """Concatenates all input AnalogSignals into a single signal ordered by t_start."""

    supported_data_object_types = [AnalogSignal]

    @property
    def filter_criteria_type(self) -> type[ConcatenationStrategyFilterCriteria]:
        """Return the FilterCriteria subclass for this strategy."""
        return ConcatenationStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[ConcatenationStrategyProxyInput]:
        """Return the ProxyInput dataclass for this strategy."""
        return ConcatenationStrategyProxyInput

    @property
    def data_input_type(self) -> type[ConcatenationStrategyDataInput]:
        """Return the DataInput dataclass for this strategy."""
        return ConcatenationStrategyDataInput

    def _infer_annotations(self, proxies: list[SignalProxy]) -> dict[str, Any]:
        """Build output annotations from the intersection of all proxies' annotations.

        Parameters
        ----------
        proxies : list[SignalProxy]
            Source proxies whose annotations are intersected.

        Returns
        -------
        dict[str, Any]
            Annotations common to all proxies, plus "concatenated": True.
        """
        proxies_annotations: list[set[tuple[str, Any]]] = []
        for proxy in proxies:
            proxies_annotations.append(set(proxy.annotations.items()))
        common_annotations: dict[str, Any] = dict(
            set.intersection(*proxies_annotations)
        )
        return {**common_annotations, "concatenated": True}

    def infer_execution_plan(
        self, input_proxies: ConcatenationStrategyProxyInput
    ) -> list[NodeDefinition]:
        """Build a single NodeDefinition that groups all input proxies.

        Parameters
        ----------
        input_proxies : ConcatenationStrategyProxyInput
            Proxies selected for this strategy run.

        Returns
        -------
        list[NodeDefinition]
            A single NodeDefinition covering all inputs.
        """
        return [
            NodeDefinition(
                input_proxies_dict=asdict(input_proxies),
                output_annotations=[self._infer_annotations(input_proxies.inputs)],
            )
        ]

    def apply(self, input_data: ConcatenationStrategyDataInput) -> list[DataObject]:
        """Concatenate all input signals into one, ordered by t_start.

        Parameters
        ----------
        input_data : ConcatenationStrategyDataInput
            Loaded data objects for one execution node.

        Returns
        -------
        list[DataObject]
            A single-element list containing the concatenated signal.
        """
        self.validate_data_objects(input_data.inputs)

        ordered_signals = sorted(input_data.inputs, key=lambda signal: signal.t_start)
        first_signal: AnalogSignal = ordered_signals[0]
        concatenated_signal = first_signal.concatenate(*ordered_signals[1:])
        return [concatenated_signal]
