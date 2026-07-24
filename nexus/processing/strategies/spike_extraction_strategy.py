"""Processing strategy that extracts spike trains from analog signals via threshold."""

from dataclasses import dataclass, asdict
from typing import Any

from elephant.spike_train_generation import threshold_detection
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
    """Filter criteria for selecting input signals for spike extraction."""

    inputs: Criteria | None = None


@dataclass
class SpikeExtractionStrategyProxyInput:
    """Input container holding SignalProxy objects for spike extraction."""

    inputs: list[SignalProxy]


@dataclass
class SpikeExtractionStrategyDataInput:
    """Input container holding loaded DataObject instances for spike extraction."""

    inputs: list[DataObject]


class SpikeExtractionStrategy(
    ProcessingStrategy[
        SpikeExtractionStrategyProxyInput,
        SpikeExtractionStrategyDataInput,
    ]
):
    """Extracts a SpikeTrain from each AnalogSignal via voltage threshold crossing."""

    supported_data_object_types = [AnalogSignal]

    def __init__(
        self,
        threshold: pq.Quantity = 0.0 * pq.mV,
        above_threshold: bool = True,
    ) -> None:
        """Initialise the spike extraction parameters.

        Parameters
        ----------
        threshold : pq.Quantity
            Voltage threshold for spike detection.
        above_threshold : bool
            If True, detect crossings above the threshold; otherwise below.
        """
        self._threshold = threshold
        self._sign = "above" if above_threshold else "below"

    @property
    def filter_criteria_type(self) -> type[SpikeExtractionStrategyFilterCriteria]:
        """Return the FilterCriteria subclass for this strategy."""
        return SpikeExtractionStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[SpikeExtractionStrategyProxyInput]:
        """Return the ProxyInput dataclass for this strategy."""
        return SpikeExtractionStrategyProxyInput

    @property
    def data_input_type(self) -> type[SpikeExtractionStrategyDataInput]:
        """Return the DataInput dataclass for this strategy."""
        return SpikeExtractionStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        """Inherit proxy annotations and mark the output as a spike train.

        Parameters
        ----------
        proxy : SignalProxy
            Source proxy whose annotations are inherited.

        Returns
        -------
        dict[str, Any]
            Merged annotations for the output signal.
        """
        return {**proxy.annotations, "spike": True}

    def infer_execution_plan(
        self, input_proxies: SpikeExtractionStrategyProxyInput
    ) -> list[NodeDefinition]:
        """Build one NodeDefinition per input proxy.

        Parameters
        ----------
        input_proxies : SpikeExtractionStrategyProxyInput
            Proxies selected for this strategy run.

        Returns
        -------
        list[NodeDefinition]
            One NodeDefinition per output signal to be computed.
        """
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
        """Detect threshold crossings in a single AnalogSignal and return spike times.

        Parameters
        ----------
        signal : AnalogSignal
            Continuous voltage signal to search for spikes.

        Returns
        -------
        SpikeTrain
            Event times of detected threshold crossings.
        """
        # threshold_detection is used instead of spike_extraction because
        # spike_extraction also extracts fixed-length waveform snippets around
        # each spike. Spikes near the signal boundaries produce truncated
        # snippets, which cannot be stacked into a uniform array and raise a
        # ValueError. threshold_detection returns only spike times, which is
        # all downstream strategies (e.g. BinnedSpikeTrainStrategy) need.
        spike_train = threshold_detection(
            signal, threshold=self._threshold, sign=self._sign
        )
        spike_train.name = signal.name
        return spike_train

    def apply(self, input_data: SpikeExtractionStrategyDataInput) -> list[DataObject]:
        """Extract a spike train from each input signal.

        Parameters
        ----------
        input_data : SpikeExtractionStrategyDataInput
            Loaded data objects for one execution node.

        Returns
        -------
        list[DataObject]
            SpikeTrain objects, one per input signal.
        """
        self.validate_data_objects(input_data.inputs)

        return [self._extract_spikes(data) for data in input_data.inputs]
