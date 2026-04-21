from dataclasses import dataclass, asdict
from typing import Any

from neo.core import AnalogSignal
from neo.core.dataobject import DataObject
import numpy as np
from scipy.signal import butter, filtfilt

from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.models import NodeDefinition
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class BandpassFilterStrategyFilterCriteria(FilterCriteria):
    inputs: Criteria | None = None


@dataclass
class BandpassFilterStrategyProxyInput:
    inputs: list[SignalProxy]


@dataclass
class BandpassFilterStrategyDataInput:
    inputs: list[DataObject]


class BandpassFilterStrategy(
    ProcessingStrategy[
        BandpassFilterStrategyProxyInput,
        BandpassFilterStrategyDataInput,
    ]
):

    supported_data_object_types = [AnalogSignal]

    def __init__(self, low: int = 500, high: int = 3000, order=4) -> None:
        self._low = low
        self._high = high
        self._order = order
        self._low_norm = None
        self._high_norm = None

    @property
    def filter_criteria_type(self) -> type[BandpassFilterStrategyFilterCriteria]:
        return BandpassFilterStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[BandpassFilterStrategyProxyInput]:
        return BandpassFilterStrategyProxyInput

    @property
    def data_input_type(self) -> type[BandpassFilterStrategyDataInput]:
        return BandpassFilterStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        return {**proxy.annotations, "filtered": True}

    def infer_execution_plan(
        self, input_proxies: BandpassFilterStrategyProxyInput
    ) -> list[NodeDefinition]:
        node_definitions = []
        for proxy in input_proxies.inputs:
            node_input_proxies = BandpassFilterStrategyProxyInput(inputs=[proxy])
            node_definitions.append(
                NodeDefinition(
                    input_proxies_dict=asdict(node_input_proxies),
                    output_annotations=[self._infer_annotations(proxy)],
                )
            )
        return node_definitions

    def _get_butterworth_coefficients(self, fs: float) -> tuple[np.ndarray, np.ndarray]:
        nyquist = fs / 2
        low_norm = self._low / nyquist
        if not 0 < low_norm < 1:
            raise ValueError(
                f"`low` must be between 0 and fs/2 = {nyquist}, but got {self._low}"
            )
        high_norm = self._high / nyquist
        if not 0 < high_norm < 1:
            raise ValueError(
                f"`high` must be between 0 and fs/2 = {nyquist}, but got {self._high}"
            )
        b, a = butter(self._order, [low_norm, high_norm], btype="band")
        return b, a

    def _filter_data(
        self, b: np.ndarray, a: np.ndarray, data: AnalogSignal
    ) -> AnalogSignal:
        filtered_data_array = filtfilt(b, a, data.magnitude, axis=0)
        return AnalogSignal(
            filtered_data_array,
            units=data.units,
            sampling_rate=data.sampling_rate,
            t_start=data.t_start,
            name=data.name,
            description=data.description,
        )

    def apply(self, input_data: BandpassFilterStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(input_data.inputs)

        # Assuming all inputs have the same sampling rate
        fs = input_data.inputs[0].sampling_rate.rescale("Hz").magnitude
        b, a = self._get_butterworth_coefficients(fs)

        return [self._filter_data(b, a, data) for data in input_data.inputs]
