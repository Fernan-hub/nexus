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
class ConcatenationStrategyFilterCriteria(FilterCriteria):
    inputs: list[Criteria] | None = None


@dataclass
class ConcatenationStrategyProxyInput:
    inputs: list[SignalProxy]


@dataclass
class ConcatenationStrategyDataInput:
    inputs: list[DataObject]


class ConcatenationStrategy(
    ProcessingStrategy[
        ConcatenationStrategyProxyInput,
        ConcatenationStrategyDataInput,
    ]
):

    supported_data_object_types = [AnalogSignal]

    @property
    def filter_criteria_type(self) -> type[ConcatenationStrategyFilterCriteria]:
        return ConcatenationStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[ConcatenationStrategyProxyInput]:
        return ConcatenationStrategyProxyInput

    @property
    def data_input_type(self) -> type[ConcatenationStrategyDataInput]:
        return ConcatenationStrategyDataInput

    def defer_application(
        self,
        inputs: ConcatenationStrategyProxyInput,
        annotation_strategy: AnnotationStrategy,
    ) -> list[SignalProxy]:
        operation_node = OperationNode(inputs.inputs, self, annotation_strategy)
        return [ComputedSignalProxy(operation_node)]

    def apply(self, inputs: ConcatenationStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(inputs.inputs)

        ordered_signals = sorted(inputs.inputs, key=lambda signal: signal.t_start)
        first_signal: AnalogSignal = ordered_signals[0]
        concatenated_signal = first_signal.concatenate(*ordered_signals[1:])
        return [concatenated_signal]
