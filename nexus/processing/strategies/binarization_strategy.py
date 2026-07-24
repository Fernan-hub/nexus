"""Processing strategy that binarizes analog signals around a voltage threshold."""

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
    """Filter criteria for selecting input signals to binarize."""

    inputs: Criteria | None = None


@dataclass
class BinarizationStrategyProxyInput:
    """Input container holding SignalProxy objects for binarization."""

    inputs: list[SignalProxy]


@dataclass
class BinarizationStrategyDataInput:
    """Input container holding loaded DataObject instances for binarization."""

    inputs: list[DataObject]


class BinarizationStrategy(
    ProcessingStrategy[
        BinarizationStrategyProxyInput,
        BinarizationStrategyDataInput,
    ]
):
    """Maps each sample of an AnalogSignal to 1 if above threshold, 0 otherwise."""

    supported_data_object_types = [AnalogSignal]

    def __init__(self, threshold: pq.Quantity = 0.0 * pq.mV) -> None:
        """Initialise the binarization threshold.

        Parameters
        ----------
        threshold : pq.Quantity
            Voltage value used to split samples into 0 and 1.
        """
        self._threshold = threshold

    @property
    def filter_criteria_type(self) -> type[BinarizationStrategyFilterCriteria]:
        """Return the FilterCriteria subclass for this strategy."""
        return BinarizationStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[BinarizationStrategyProxyInput]:
        """Return the ProxyInput dataclass for this strategy."""
        return BinarizationStrategyProxyInput

    @property
    def data_input_type(self) -> type[BinarizationStrategyDataInput]:
        """Return the DataInput dataclass for this strategy."""
        return BinarizationStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        """Inherit proxy annotations and mark the output as binary.

        Parameters
        ----------
        proxy : SignalProxy
            Source proxy whose annotations are inherited.

        Returns
        -------
        dict[str, Any]
            Merged annotations for the output signal.
        """
        return {**proxy.annotations, "binary": True}

    def infer_execution_plan(
        self, input_proxies: BinarizationStrategyProxyInput
    ) -> list[NodeDefinition]:
        """Build one NodeDefinition per input proxy.

        Parameters
        ----------
        input_proxies : BinarizationStrategyProxyInput
            Proxies selected for this strategy run.

        Returns
        -------
        list[NodeDefinition]
            One NodeDefinition per output signal to be computed.
        """
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
        """Convert a single AnalogSignal to a dimensionless 0/1 signal.

        Parameters
        ----------
        data : AnalogSignal
            Signal to binarize.

        Returns
        -------
        AnalogSignal
            Dimensionless signal with values 0.0 or 1.0.
        """
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
        """Binarize all input signals independently.

        Parameters
        ----------
        input_data : BinarizationStrategyDataInput
            Loaded data objects for one execution node.

        Returns
        -------
        list[DataObject]
            Binarized analog signals.
        """
        self.validate_data_objects(input_data.inputs)

        return [self._to_binary(data) for data in input_data.inputs]
