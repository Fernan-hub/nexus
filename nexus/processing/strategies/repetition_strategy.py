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
    inputs: Criteria | None = None


@dataclass
class RepetitionStrategyProxyInput:
    inputs: list[SignalProxy]


@dataclass
class RepetitionStrategyDataInput:
    inputs: list[DataObject]


class RepetitionStrategy(
    ProcessingStrategy[
        RepetitionStrategyProxyInput,
        RepetitionStrategyDataInput,
    ]
):
    supported_data_object_types = [AnalogSignal]

    def _validate_parameters(self) -> None:
        if self._repetitions < 1:
            raise ValueError("Repetitions must be at least 1")

    def __init__(self, repetitions: int) -> None:
        self._repetitions = repetitions
        self._validate_parameters()

    @property
    def filter_criteria_type(self) -> type[RepetitionStrategyFilterCriteria]:
        return RepetitionStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[RepetitionStrategyProxyInput]:
        return RepetitionStrategyProxyInput

    @property
    def data_input_type(self) -> type[RepetitionStrategyDataInput]:
        return RepetitionStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        return {**proxy.annotations, "repeated": True}

    def infer_execution_plan(
        self, input_proxies: RepetitionStrategyProxyInput
    ) -> list[NodeDefinition]:
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
        shifted_copies = []
        previous = signal
        for _ in range(self._repetitions - 1):
            next_copy = previous.time_shift(previous.duration)
            shifted_copies.append(next_copy)
            previous = next_copy

        return signal.concatenate(*shifted_copies)

    def apply(self, input_data: RepetitionStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(input_data.inputs)

        return [self._concatenate_repetitions(data) for data in input_data.inputs]
