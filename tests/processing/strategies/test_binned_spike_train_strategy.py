"""Tests for BinnedSpikeTrainStrategy."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal, SpikeTrain

from nexus.core.interfaces import SignalProxy
from nexus.processing.strategies.binned_spike_train_strategy import (
    BinnedSpikeTrainStrategy,
    BinnedSpikeTrainStrategyDataInput,
    BinnedSpikeTrainStrategyProxyInput,
)


class TestBinnedSpikeTrainStrategy(unittest.TestCase):
    """Unit tests for BinnedSpikeTrainStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_infer_execution_plan(self) -> None:
        """Test that each proxy gets its own NodeDefinition with its annotations plus binned=True."""
        proxy1 = MagicMock(spec=SignalProxy)
        proxy1.annotations = {"model": "efish", "spike": True}
        proxy2 = MagicMock(spec=SignalProxy)
        proxy2.annotations = {"channel": "B"}

        strategy = BinnedSpikeTrainStrategy(bin_size=1 * pq.ms)
        input_proxies = BinnedSpikeTrainStrategyProxyInput(inputs=[proxy1, proxy2])

        result = strategy.infer_execution_plan(input_proxies)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].input_proxies_dict["inputs"]), 1)
        self.assertEqual(len(result[1].input_proxies_dict["inputs"]), 1)
        self.assertEqual(
            result[0].output_annotations[0],
            {"model": "efish", "spike": True, "binned": True},
        )
        self.assertEqual(
            result[1].output_annotations[0], {"channel": "B", "binned": True}
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.processing.strategies.binned_spike_train_strategy.BinnedSpikeTrain")
    def test_apply(self, mock_bst_class: MagicMock) -> None:
        """Test that apply returns AnalogSignal with binary values derived from BinnedSpikeTrain."""
        mock_bst_instance = MagicMock()
        mock_bst_instance.to_bool_array.return_value = np.array(
            [[True, False, True, False]]
        )
        mock_bst_class.return_value = mock_bst_instance

        spike_train = SpikeTrain(
            np.array([0.25, 0.75]) * pq.s,
            t_stop=2.0 * pq.s,
            t_start=0.0 * pq.s,
            name="spikes",
        )
        strategy = BinnedSpikeTrainStrategy(bin_size=0.5 * pq.s)
        input_data = BinnedSpikeTrainStrategyDataInput(inputs=[spike_train])

        result = strategy.apply(input_data)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AnalogSignal)
        self.assertEqual(result[0].units, pq.dimensionless)
        self.assertEqual(result[0].name, "spikes")
        np.testing.assert_array_equal(result[0].magnitude.flatten(), [1, 0, 1, 0])
        mock_bst_class.assert_called_once()
        mock_bst_instance.to_bool_array.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_apply_raises_type_error_when_input_is_not_spike_train(self) -> None:
        """Test that apply raises TypeError when an AnalogSignal is passed instead of SpikeTrain."""
        signal = AnalogSignal(
            np.array([0.0, 1.0]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
        )
        strategy = BinnedSpikeTrainStrategy()
        input_data = BinnedSpikeTrainStrategyDataInput(inputs=[signal])

        with self.assertRaises(TypeError):
            strategy.apply(input_data)


class TestIntegrationBinnedSpikeTrainStrategy(unittest.TestCase):
    """Integration tests for BinnedSpikeTrainStrategy with real elephant binning."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply(self) -> None:
        """Test that apply produces a binary AnalogSignal with 1 in bins that contain a spike."""
        spike_train = SpikeTrain(
            np.array([0.25]) * pq.s,  # one spike at 0.25 s, inside [0, 0.5) bin
            t_stop=1.0 * pq.s,
            t_start=0.0 * pq.s,
            name="spikes",
        )
        strategy = BinnedSpikeTrainStrategy(bin_size=0.5 * pq.s)
        input_data = BinnedSpikeTrainStrategyDataInput(inputs=[spike_train])

        result = strategy.apply(input_data)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AnalogSignal)
        self.assertEqual(len(result[0]), 2)  # 2 bins of 0.5 s in a 1 s window
        self.assertEqual(result[0].units, pq.dimensionless)
        self.assertEqual(result[0].name, "spikes")
        np.testing.assert_array_equal(result[0].magnitude.flatten(), [1, 0])

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply_sampling_rate_equals_inverse_bin_size(self) -> None:
        """Test that the output AnalogSignal sampling rate equals 1 / bin_size."""
        spike_train = SpikeTrain(
            np.array([0.1]) * pq.s,
            t_stop=1.0 * pq.s,
            t_start=0.0 * pq.s,
        )
        bin_size = 10.0 * pq.ms
        strategy = BinnedSpikeTrainStrategy(bin_size=bin_size)
        input_data = BinnedSpikeTrainStrategyDataInput(inputs=[spike_train])

        result = strategy.apply(input_data)

        expected_sampling_rate = (1.0 / bin_size).rescale(pq.Hz)
        self.assertAlmostEqual(
            float(result[0].sampling_rate.rescale(pq.Hz)),
            float(expected_sampling_rate),
            places=6,
        )
