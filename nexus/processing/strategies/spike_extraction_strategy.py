from dataclasses import dataclass, asdict
from typing import Any

from elephant.spike_train_generation import spike_extraction
from neo.core import AnalogSignal, SpikeTrain
from neo.core.dataobject import DataObject

import quantities as pq

from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.models import NodeDefinition
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class SpikeExtractionStrategyFilterCriteria(FilterCriteria):
    inputs: Criteria | None = None


@dataclass
class SpikeExtractionStrategyProxyInput:
    inputs: list[SignalProxy]


@dataclass
class SpikeExtractionStrategyDataInput:
    inputs: list[DataObject]


class SpikeExtractionStrategy(
    ProcessingStrategy[
        SpikeExtractionStrategyProxyInput,
        SpikeExtractionStrategyDataInput,
    ]
):

    supported_data_object_types = [AnalogSignal]

    def __init__(
        self,
        threshold: pq.Quantity = 0.0 * pq.mV,
        above_threshold: bool = True,
    ) -> None:
        self._threshold = threshold
        self._sign = "above" if above_threshold else "below"

    @property
    def filter_criteria_type(self) -> type[SpikeExtractionStrategyFilterCriteria]:
        return SpikeExtractionStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[SpikeExtractionStrategyProxyInput]:
        return SpikeExtractionStrategyProxyInput

    @property
    def data_input_type(self) -> type[SpikeExtractionStrategyDataInput]:
        return SpikeExtractionStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        return {**proxy.annotations, "spike": True}

    def infer_execution_plan(
        self, input_proxies: SpikeExtractionStrategyProxyInput
    ) -> list[NodeDefinition]:
        node_definitions = []
        for proxy in input_proxies.inputs:
            node_input_proxies = SpikeExtractionStrategyProxyInput(inputs=[proxy])
            node_definitions.append(
                NodeDefinition(
                    input_proxies_dict=asdict(node_input_proxies),
                    output_annotations=[self._infer_annotations(proxy)],
                )
            )
        return node_definitions

    def _extract_spikes(self, signal: AnalogSignal) -> SpikeTrain:
        return spike_extraction(signal, threshold=self._threshold, sign=self._sign)

    def apply(self, input_data: SpikeExtractionStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(input_data.inputs)

        return [self._extract_spikes(data) for data in input_data.inputs]
