"""Processing strategy that zero-mean, unit-variance standardizes analog signals."""

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
    """Filter criteria for selecting input signals to standardize."""

    inputs: Criteria | None = None


@dataclass
class StandardizationStrategyProxyInput:
    """Input container holding SignalProxy objects for standardization."""

    inputs: list[SignalProxy]


@dataclass
class StandardizationStrategyDataInput:
    """Input container holding loaded DataObject instances for standardization."""

    inputs: list[DataObject]


class StandardizationStrategy(
    ProcessingStrategy[
        StandardizationStrategyProxyInput,
        StandardizationStrategyDataInput,
    ]
):
    """Standardizes each AnalogSignal to zero mean and unit variance."""

    supported_data_object_types = [AnalogSignal]

    @property
    def filter_criteria_type(self) -> type[StandardizationStrategyFilterCriteria]:
        """Return the FilterCriteria subclass for this strategy."""
        return StandardizationStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[StandardizationStrategyProxyInput]:
        """Return the ProxyInput dataclass for this strategy."""
        return StandardizationStrategyProxyInput

    @property
    def data_input_type(self) -> type[StandardizationStrategyDataInput]:
        """Return the DataInput dataclass for this strategy."""
        return StandardizationStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        """Inherit proxy annotations and mark the output as standardized.

        Parameters
        ----------
        proxy : SignalProxy
            Source proxy whose annotations are inherited.

        Returns
        -------
        dict[str, Any]
            Merged annotations for the output signal.
        """
        return {**proxy.annotations, "standardized": True}

    def infer_execution_plan(
        self, input_proxies: StandardizationStrategyProxyInput
    ) -> list[NodeDefinition]:
        """Build one NodeDefinition per input proxy.

        Parameters
        ----------
        input_proxies : StandardizationStrategyProxyInput
            Proxies selected for this strategy run.

        Returns
        -------
        list[NodeDefinition]
            One NodeDefinition per output signal to be computed.
        """
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
        """Subtract the mean and divide by the standard deviation.

        Parameters
        ----------
        data : DataObject
            Signal to standardize.

        Returns
        -------
        DataObject
            Standardized signal preserving the original units.
        """
        # Divide by std magnitude to preserve units
        return (data - np.mean(data)) / np.std(data).magnitude

    def apply(self, input_data: StandardizationStrategyDataInput) -> list[DataObject]:
        """Standardize all input signals independently.

        Parameters
        ----------
        input_data : StandardizationStrategyDataInput
            Loaded data objects for one execution node.

        Returns
        -------
        list[DataObject]
            Standardized analog signals.
        """
        self.validate_data_objects(input_data.inputs)

        return [self._standardize(data) for data in input_data.inputs]
