"""Processing strategy that converts spike trains to binned analog signals."""

from dataclasses import dataclass, asdict
from typing import Any

from elephant.conversion import BinnedSpikeTrain
from neo.core import AnalogSignal, SpikeTrain
from neo.core.dataobject import DataObject

import numpy as np
import quantities as pq

from nexus.common.interfaces import FilterCriteria
from nexus.core.interfaces import SignalProxy
from nexus.models import NodeDefinition
from nexus.processing.interfaces import ProcessingStrategy
from nexus.types import Criteria


@dataclass
class BinnedSpikeTrainStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting spike train inputs to bin."""

    inputs: Criteria | None = None


@dataclass
class BinnedSpikeTrainStrategyProxyInput:
    """Input container holding SignalProxy objects for spike train binning."""

    inputs: list[SignalProxy]


@dataclass
class BinnedSpikeTrainStrategyDataInput:
    """Input container holding loaded DataObject instances for spike train binning."""

    inputs: list[DataObject]


class BinnedSpikeTrainStrategy(
    ProcessingStrategy[
        BinnedSpikeTrainStrategyProxyInput,
        BinnedSpikeTrainStrategyDataInput,
    ]
):
    """Bins each SpikeTrain into a boolean AnalogSignal of fixed-width time bins."""

    supported_data_object_types = [SpikeTrain]

    def __init__(
        self,
        bin_size: pq.Quantity = 1 * pq.ms,
        t_start: pq.Quantity | None = None,
        t_stop: pq.Quantity | None = None,
        tolerance: float = 1e-8,
    ) -> None:
        """Initialise the binning parameters.

        Parameters
        ----------
        bin_size : pq.Quantity
            Width of each time bin.
        t_start : pq.Quantity or None
            Start of the binning window; defaults to the spike train's t_start.
        t_stop : pq.Quantity or None
            End of the binning window; defaults to the spike train's t_stop.
        tolerance : float
            Floating-point tolerance passed to BinnedSpikeTrain.
        """
        self._bin_size = bin_size
        self._t_start = t_start
        self._t_stop = t_stop
        self._tolerance = tolerance

    @property
    def filter_criteria_type(self) -> type[BinnedSpikeTrainStrategyFilterCriteria]:
        """Return the FilterCriteria subclass for this strategy."""
        return BinnedSpikeTrainStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[BinnedSpikeTrainStrategyProxyInput]:
        """Return the ProxyInput dataclass for this strategy."""
        return BinnedSpikeTrainStrategyProxyInput

    @property
    def data_input_type(self) -> type[BinnedSpikeTrainStrategyDataInput]:
        """Return the DataInput dataclass for this strategy."""
        return BinnedSpikeTrainStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        """Inherit proxy annotations and mark the output as binned.

        Parameters
        ----------
        proxy : SignalProxy
            Source proxy whose annotations are inherited.

        Returns
        -------
        dict[str, Any]
            Merged annotations for the output signal.
        """
        return {**proxy.annotations, "binned": True}

    def infer_execution_plan(
        self, input_proxies: BinnedSpikeTrainStrategyProxyInput
    ) -> list[NodeDefinition]:
        """Build one NodeDefinition per input proxy.

        Parameters
        ----------
        input_proxies : BinnedSpikeTrainStrategyProxyInput
            Proxies selected for this strategy run.

        Returns
        -------
        list[NodeDefinition]
            One NodeDefinition per output signal to be computed.
        """
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
        """Convert a single SpikeTrain to a boolean AnalogSignal.

        Parameters
        ----------
        spike_train : SpikeTrain
            Spike times to bin.

        Returns
        -------
        AnalogSignal
            Dimensionless int32 signal where 1 indicates a spike in that bin.
        """
        bst = BinnedSpikeTrain(
            spike_train,
            bin_size=self._bin_size,
            t_start=self._t_start,
            t_stop=self._t_stop,
            tolerance=self._tolerance,
        )
        # Cast to int32: AnalogSignal stores float64 by default, and
        # DiscreteMIEstimator warns when it receives float arrays because it
        # expects properly symbolized (integer) data.
        binary_array = bst.to_bool_array()[0].astype(np.int32)
        sampling_rate = (1.0 / self._bin_size).rescale(pq.Hz)
        return AnalogSignal(
            signal=binary_array,
            units=pq.dimensionless,
            sampling_rate=sampling_rate,
            t_start=spike_train.t_start,
            name=spike_train.name,
        )

    def apply(self, input_data: BinnedSpikeTrainStrategyDataInput) -> list[DataObject]:
        """Bin each input spike train into an AnalogSignal.

        Parameters
        ----------
        input_data : BinnedSpikeTrainStrategyDataInput
            Loaded data objects for one execution node.

        Returns
        -------
        list[DataObject]
            Binned analog signals, one per input spike train.
        """
        self.validate_data_objects(input_data.inputs)
        return [self._bin_spike_train(data) for data in input_data.inputs]
