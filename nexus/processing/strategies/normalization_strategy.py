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
class NormalizationStrategyFilterCriteria(FilterCriteria):
    inputs: Criteria | None = None


@dataclass
class NormalizationStrategyProxyInput:
    inputs: list[SignalProxy]


@dataclass
class NormalizationStrategyDataInput:
    inputs: list[DataObject]


class NormalizationStrategy(
    ProcessingStrategy[
        NormalizationStrategyProxyInput,
        NormalizationStrategyDataInput,
    ]
):

    supported_data_object_types = [AnalogSignal]

    def __init__(
        self,
        low: pq.Quantity = 0 * pq.mV,
        high: pq.Quantity = 1 * pq.mV,
    ) -> None:
        self._low = low
        self._high = high

    @property
    def filter_criteria_type(self) -> type[NormalizationStrategyFilterCriteria]:
        return NormalizationStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[NormalizationStrategyProxyInput]:
        return NormalizationStrategyProxyInput

    @property
    def data_input_type(self) -> type[NormalizationStrategyDataInput]:
        return NormalizationStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        return {**proxy.annotations, "normalized": True}

    def infer_execution_plan(
        self, input_proxies: NormalizationStrategyProxyInput
    ) -> list[NodeDefinition]:
        node_definitions = []
        for proxy in input_proxies.inputs:
            node_input_proxies = NormalizationStrategyProxyInput(inputs=[proxy])
            node_definitions.append(
                NodeDefinition(
                    input_proxies_dict=asdict(node_input_proxies),
                    output_annotations=[self._infer_annotations(proxy)],
                )
            )
        return node_definitions

    def _normalize(self, data: AnalogSignal) -> AnalogSignal:
        values = data.magnitude
        min_val = values.min(axis=0, keepdims=True)
        max_val = values.max(axis=0, keepdims=True)
        data_range = np.where(max_val - min_val == 0, 1, max_val - min_val)
        low = self._low.rescale(self._high.units).magnitude
        high = self._high.magnitude
        scaled = (values - min_val) / data_range * (high - low) + low
        return AnalogSignal(
            scaled,
            units=self._high.units,
            sampling_rate=data.sampling_rate,
            t_start=data.t_start,
            name=data.name,
            description=data.description,
        )

    def apply(self, input_data: NormalizationStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(input_data.inputs)

        return [self._normalize(data) for data in input_data.inputs]
