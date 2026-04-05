from dataclasses import dataclass

from neo.core import AnalogSignal
from neo.core.dataobject import DataObject

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.core.operation_node import OperationNode
from nexus.core.proxies import ComputedSignalProxy
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class RepetitionStrategyFilterCriteria(FilterCriteria):
    inputs: list[Criteria] | None = None


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

    def defer_application(
        self,
        inputs: RepetitionStrategyProxyInput,
        annotation_strategy: AnnotationStrategy,
    ) -> list[SignalProxy]:
        return [
            ComputedSignalProxy(OperationNode([proxy], self, annotation_strategy))
            for proxy in inputs.inputs
        ]

    def _concatenate_repetitions(self, signal: AnalogSignal) -> AnalogSignal:
        shifted_copies = []
        previous = signal
        for _ in range(self._repetitions - 1):
            next_copy = previous.time_shift(previous.duration)
            shifted_copies.append(next_copy)
            previous = next_copy

        return signal.concatenate(*shifted_copies)

    def apply(self, inputs: RepetitionStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(inputs.inputs)

        return [self._concatenate_repetitions(data) for data in inputs.inputs]
