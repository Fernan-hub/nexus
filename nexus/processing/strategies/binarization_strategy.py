from dataclasses import dataclass, asdict
from typing import Any

from neo.core import AnalogSignal
from neo.core.dataobject import DataObject
import numpy as np
import quantities as pq

from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.models import NodeDefinition
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class BinarizationStrategyFilterCriteria(FilterCriteria):
    inputs: Criteria | None = None


@dataclass
class BinarizationStrategyProxyInput:
    inputs: list[SignalProxy]


@dataclass
class BinarizationStrategyDataInput:
    inputs: list[DataObject]


class BinarizationStrategy(
    ProcessingStrategy[
        BinarizationStrategyProxyInput,
        BinarizationStrategyDataInput,
    ]
):

    supported_data_object_types = [AnalogSignal]

    def __init__(self, threshold: pq.Quantity = 0.0 * pq.mV) -> None:
        self._threshold = threshold

    @property
    def filter_criteria_type(self) -> type[BinarizationStrategyFilterCriteria]:
        return BinarizationStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[BinarizationStrategyProxyInput]:
        return BinarizationStrategyProxyInput

    @property
    def data_input_type(self) -> type[BinarizationStrategyDataInput]:
        return BinarizationStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        return {**proxy.annotations, "binary": True}

    def infer_execution_plan(
        self, input_proxies: BinarizationStrategyProxyInput
    ) -> list[NodeDefinition]:
        node_definitions = []
        for proxy in input_proxies.inputs:
            node_input_proxies = BinarizationStrategyProxyInput([proxy])
            node_definitions.append(
                NodeDefinition(
                    input_proxies_dict=asdict(node_input_proxies),
                    output_annotations=[self._infer_annotations(proxy)],
                )
            )
        return node_definitions

    def _to_binary(self, data: AnalogSignal) -> AnalogSignal:
        threshold = self._threshold.rescale(data.units).magnitude
        binary = np.where(data.magnitude > threshold, 1.0, 0.0)
        return AnalogSignal(
            binary,
            units=pq.dimensionless,
            sampling_rate=data.sampling_rate,
            t_start=data.t_start,
            name=data.name,
            description=data.description,
        )

    def apply(self, input_data: BinarizationStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(input_data.inputs)

        return [self._to_binary(data) for data in input_data.inputs]
