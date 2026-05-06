from dataclasses import dataclass, asdict
from typing import Any

from elephant.conversion import BinnedSpikeTrain
from neo.core import AnalogSignal, SpikeTrain
from neo.core.dataobject import DataObject

import quantities as pq

from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.models import NodeDefinition
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class BinnedSpikeTrainStrategyFilterCriteria(FilterCriteria):
    inputs: Criteria | None = None


@dataclass
class BinnedSpikeTrainStrategyProxyInput:
    inputs: list[SignalProxy]


@dataclass
class BinnedSpikeTrainStrategyDataInput:
    inputs: list[DataObject]


class BinnedSpikeTrainStrategy(
    ProcessingStrategy[
        BinnedSpikeTrainStrategyProxyInput,
        BinnedSpikeTrainStrategyDataInput,
    ]
):

    supported_data_object_types = [SpikeTrain]

    def __init__(self, bin_size: pq.Quantity = 1 * pq.ms) -> None:
        self._bin_size = bin_size

    @property
    def filter_criteria_type(self) -> type[BinnedSpikeTrainStrategyFilterCriteria]:
        return BinnedSpikeTrainStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[BinnedSpikeTrainStrategyProxyInput]:
        return BinnedSpikeTrainStrategyProxyInput

    @property
    def data_input_type(self) -> type[BinnedSpikeTrainStrategyDataInput]:
        return BinnedSpikeTrainStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        return {**proxy.annotations, "binned": True}

    def infer_execution_plan(
        self, input_proxies: BinnedSpikeTrainStrategyProxyInput
    ) -> list[NodeDefinition]:
        node_definitions = []
        for proxy in input_proxies.inputs:
            node_input_proxies = BinnedSpikeTrainStrategyProxyInput(inputs=[proxy])
            node_definitions.append(
                NodeDefinition(
                    input_proxies_dict=asdict(node_input_proxies),
                    output_annotations=[self._infer_annotations(proxy)],
                )
            )
        return node_definitions

    def _bin_spike_train(self, spike_train: SpikeTrain) -> AnalogSignal:
        bst = BinnedSpikeTrain(spike_train, bin_size=self._bin_size)
        binary_array = bst.to_array(binary=True)[0].astype(float)
        sampling_rate = (1.0 / self._bin_size).rescale(pq.Hz)
        return AnalogSignal(
            signal=binary_array,
            units=pq.dimensionless,
            sampling_rate=sampling_rate,
            t_start=spike_train.t_start,
            name=spike_train.name,
        )

    def apply(self, input_data: BinnedSpikeTrainStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(input_data.inputs)
        return [self._bin_spike_train(data) for data in input_data.inputs]
