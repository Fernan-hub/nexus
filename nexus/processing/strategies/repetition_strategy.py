"""Processing strategy that concatenates N time-shifted copies of each signal."""

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
class RepetitionStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting input signals to repeat."""

    inputs: Criteria | None = None


@dataclass
class RepetitionStrategyProxyInput:
    """Input container holding SignalProxy objects for repetition."""

    inputs: list[SignalProxy]


@dataclass
class RepetitionStrategyDataInput:
    """Input container holding loaded DataObject instances for repetition."""

    inputs: list[DataObject]


class RepetitionStrategy(
    ProcessingStrategy[
        RepetitionStrategyProxyInput,
        RepetitionStrategyDataInput,
    ]
):
    """Produces a signal made of N back-to-back copies of each input AnalogSignal."""

    supported_data_object_types = [AnalogSignal]

    def _validate_parameters(self) -> None:
        if self._repetitions < 1:
            raise ValueError("Repetitions must be at least 1")

    def __init__(self, repetitions: int) -> None:
        """Initialise the repetition count.

        Parameters
        ----------
        repetitions : int
            Number of times each signal is repeated; must be at least 1.
        """
        self._repetitions = repetitions
        self._validate_parameters()

    @property
    def filter_criteria_type(self) -> type[RepetitionStrategyFilterCriteria]:
        """Return the FilterCriteria subclass for this strategy."""
        return RepetitionStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[RepetitionStrategyProxyInput]:
        """Return the ProxyInput dataclass for this strategy."""
        return RepetitionStrategyProxyInput

    @property
    def data_input_type(self) -> type[RepetitionStrategyDataInput]:
        """Return the DataInput dataclass for this strategy."""
        return RepetitionStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        """Inherit proxy annotations and mark the output as repeated.

        Parameters
        ----------
        proxy : SignalProxy
            Source proxy whose annotations are inherited.

        Returns
        -------
        dict[str, Any]
            Merged annotations for the output signal.
        """
        return {**proxy.annotations, "repeated": True}

    def infer_execution_plan(
        self, input_proxies: RepetitionStrategyProxyInput
    ) -> list[NodeDefinition]:
        """Build one NodeDefinition per input proxy.

        Parameters
        ----------
        input_proxies : RepetitionStrategyProxyInput
            Proxies selected for this strategy run.

        Returns
        -------
        list[NodeDefinition]
            One NodeDefinition per output signal to be computed.
        """
        node_definitions = []
        for proxy in input_proxies.inputs:
            node_input_proxies = RepetitionStrategyProxyInput(inputs=[proxy])
            node_definitions.append(
                NodeDefinition(
                    input_proxies_dict=asdict(node_input_proxies),
                    output_annotations=[self._infer_annotations(proxy)],
                )
            )
        return node_definitions

    def _concatenate_repetitions(self, signal: AnalogSignal) -> AnalogSignal:
        """Build N back-to-back time-shifted copies of a signal and concatenate them.

        Parameters
        ----------
        signal : AnalogSignal
            Signal to repeat.

        Returns
        -------
        AnalogSignal
            Signal of duration N * signal.duration.
        """
        shifted_copies = []
        previous = signal
        for _ in range(self._repetitions - 1):
            next_copy = previous.time_shift(previous.duration)
            shifted_copies.append(next_copy)
            previous = next_copy

        return signal.concatenate(*shifted_copies)

    def apply(self, input_data: RepetitionStrategyDataInput) -> list[DataObject]:
        """Repeat each input signal N times.

        Parameters
        ----------
        input_data : RepetitionStrategyDataInput
            Loaded data objects for one execution node.

        Returns
        -------
        list[DataObject]
            Repeated analog signals, one per input.
        """
        self.validate_data_objects(input_data.inputs)

        return [self._concatenate_repetitions(data) for data in input_data.inputs]
