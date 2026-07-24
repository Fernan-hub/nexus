"""Processing strategy that applies a Butterworth bandpass filter to analog signals."""

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
    """Filter criteria for selecting input signals to bandpass-filter."""

    inputs: Criteria | None = None


@dataclass
class BandpassFilterStrategyProxyInput:
    """Input container holding SignalProxy objects for the bandpass filter."""

    inputs: list[SignalProxy]


@dataclass
class BandpassFilterStrategyDataInput:
    """Input container holding loaded DataObject instances for the bandpass filter."""

    inputs: list[DataObject]


class BandpassFilterStrategy(
    ProcessingStrategy[
        BandpassFilterStrategyProxyInput,
        BandpassFilterStrategyDataInput,
    ]
):
    """Applies a zero-phase Butterworth bandpass filter to each input AnalogSignal."""

    supported_data_object_types = [AnalogSignal]

    def __init__(self, low: float = 500, high: float = 3000, order: int = 4) -> None:
        """Initialise the bandpass filter parameters.

        Parameters
        ----------
        low : float
            Lower cutoff frequency in Hz.
        high : float
            Upper cutoff frequency in Hz.
        order : int
            Butterworth filter order.
        """
        self._low = low
        self._high = high
        self._order = order
        self._low_norm = None
        self._high_norm = None

    @property
    def filter_criteria_type(self) -> type[BandpassFilterStrategyFilterCriteria]:
        """Return the FilterCriteria subclass for this strategy."""
        return BandpassFilterStrategyFilterCriteria

    @property
    def proxy_input_type(self) -> type[BandpassFilterStrategyProxyInput]:
        """Return the ProxyInput dataclass for this strategy."""
        return BandpassFilterStrategyProxyInput

    @property
    def data_input_type(self) -> type[BandpassFilterStrategyDataInput]:
        """Return the DataInput dataclass for this strategy."""
        return BandpassFilterStrategyDataInput

    def _infer_annotations(self, proxy: SignalProxy) -> dict[str, Any]:
        """Inherit proxy annotations and mark the output as filtered.

        Parameters
        ----------
        proxy : SignalProxy
            Source proxy whose annotations are inherited.

        Returns
        -------
        dict[str, Any]
            Merged annotations for the output signal.
        """
        return {**proxy.annotations, "filtered": True}

    def infer_execution_plan(
        self, input_proxies: BandpassFilterStrategyProxyInput
    ) -> list[NodeDefinition]:
        """Build one NodeDefinition per input proxy.

        Parameters
        ----------
        input_proxies : BandpassFilterStrategyProxyInput
            Proxies selected for this strategy run.

        Returns
        -------
        list[NodeDefinition]
            One NodeDefinition per output signal to be computed.
        """
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
        """Compute Butterworth bandpass coefficients normalised to [0, 1].

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Numerator (b) and denominator (a) filter coefficient arrays.
        """
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
        """Apply zero-phase filtering to a single AnalogSignal.

        Parameters
        ----------
        b : np.ndarray
            Numerator filter coefficients.
        a : np.ndarray
            Denominator filter coefficients.
        data : AnalogSignal
            Signal to filter.

        Returns
        -------
        AnalogSignal
            Filtered signal with the same units, sampling rate, and t_start.
        """
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
        """Filter all input signals with a shared Butterworth bandpass filter.

        Parameters
        ----------
        input_data : BandpassFilterStrategyDataInput
            Loaded data objects for one execution node.

        Returns
        -------
        list[DataObject]
            Bandpass-filtered analog signals.
        """
        self.validate_data_objects(input_data.inputs)

        # Assuming all inputs have the same sampling rate
        fs = input_data.inputs[0].sampling_rate.rescale("Hz").magnitude
        b, a = self._get_butterworth_coefficients(fs)

        return [self._filter_data(b, a, data) for data in input_data.inputs]
