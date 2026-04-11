from dataclasses import dataclass, asdict

from neo.core import AnalogSignal
from neo.core.dataobject import DataObject
import numpy as np

from nexus.annotation.interfaces import AnnotationStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.core.operation_node import OperationNode
from nexus.core.proxies import ComputedSignalProxy
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class StandardizationStrategyFilterCriteria(FilterCriteria):
    inputs: list[Criteria] | None = None


@dataclass
class StandardizationStrategyProxyInput:
    inputs: list[SignalProxy]


@dataclass
class StandardizationStrategyDataInput:
    inputs: list[DataObject]


class StandardizationStrategy(
    ProcessingStrategy[
        StandardizationStrategyProxyInput,
        StandardizationStrategyDataInput,
    ]
):

    supported_data_object_types = [AnalogSignal]

    @property
    def filter_criteria_type(self) -> type[StandardizationStrategyFilterCriteria]:
        return StandardizationStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[StandardizationStrategyProxyInput]:
        return StandardizationStrategyProxyInput

    @property
    def data_input_type(self) -> type[StandardizationStrategyDataInput]:
        return StandardizationStrategyDataInput

    def defer_application(
        self,
        inputs: StandardizationStrategyProxyInput,
        annotation_strategy: AnnotationStrategy,
    ) -> list[SignalProxy]:
        output_proxies = []
        for proxy in inputs.inputs:
            parent_proxies = StandardizationStrategyProxyInput(inputs=[proxy])
            operation_node = OperationNode(
                asdict(parent_proxies), self, annotation_strategy
            )
            output_proxies.append(ComputedSignalProxy(operation_node))
        return output_proxies

    def _standardize(self, data: DataObject) -> DataObject:
        # Divide by std magnitude to preserve units
        return (data - np.mean(data)) / np.std(data).magnitude

    def apply(self, inputs: StandardizationStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(inputs.inputs)

        return [self._standardize(data) for data in inputs.inputs]
