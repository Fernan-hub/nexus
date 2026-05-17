from dataclasses import dataclass, asdict
from typing import Any

from neo.core import AnalogSignal
from neo.core.dataobject import DataObject
import numpy as np

from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.models import NodeDefinition
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class StandardizationStrategyFilterCriteria(FilterCriteria):
    inputs: Criteria | None = None


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

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        return {**proxy.annotations, "standardized": True}

    def infer_execution_plan(
        self, input_proxies: StandardizationStrategyProxyInput
    ) -> list[NodeDefinition]:
        node_definitions = []
        for proxy in input_proxies.inputs:
            node_input_proxies = StandardizationStrategyProxyInput([proxy])
            node_definitions.append(
                NodeDefinition(
                    input_proxies_dict=asdict(node_input_proxies),
                    output_annotations=[self._infer_annotations(proxy)],
                )
            )
        return node_definitions

    def _standardize(self, data: DataObject) -> DataObject:
        # Divide by std magnitude to preserve units
        return (data - np.mean(data)) / np.std(data).magnitude

    def apply(self, input_data: StandardizationStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(input_data.inputs)

        return [self._standardize(data) for data in input_data.inputs]
