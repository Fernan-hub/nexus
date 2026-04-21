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
class ConcatenationStrategyFilterCriteria(FilterCriteria):
    inputs: Criteria | None = None


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

    def _infer_annotations(self, proxies: list[SignalProxy]) -> dict[str, Any]:
        proxies_annotations: list[set[tuple[str, Any]]] = []
        for proxy in proxies:
            proxies_annotations.append(set(proxy.annotations.items()))
        common_annotations: dict[str, Any] = dict(
            set.intersection(*proxies_annotations)
        )
        return {**common_annotations, "concatenated": True}

    def infer_execution_plan(
        self, input_proxies: ConcatenationStrategyProxyInput
    ) -> list[NodeDefinition]:
        return [
            NodeDefinition(
                input_proxies_dict=asdict(input_proxies),
                output_annotations=self._infer_annotations(input_proxies.inputs),
            )
        ]

    def apply(self, input_data: ConcatenationStrategyDataInput) -> list[DataObject]:
        self.validate_data_objects(input_data.inputs)

        ordered_signals = sorted(input_data.inputs, key=lambda signal: signal.t_start)
        first_signal: AnalogSignal = ordered_signals[0]
        concatenated_signal = first_signal.concatenate(*ordered_signals[1:])
        return [concatenated_signal]
