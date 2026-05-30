"""Tests for BandpassFilterStrategy."""

import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal, SpikeTrain

from nexus.core.interfaces import SignalProxy
from nexus.processing.strategies.bandpass_filter_strategy import (
    BandpassFilterStrategy,
    BandpassFilterStrategyDataInput,
    BandpassFilterStrategyFilterCriteria,
    BandpassFilterStrategyProxyInput,
)
from tests.utils import Expected, Given, Scenario


class TestBandpassFilterStrategy(unittest.TestCase):
    """Unit tests for BandpassFilterStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_infer_execution_plan(self) -> None:
        """Test that each proxy gets its own NodeDefinition with its annotations plus filtered=True."""
        proxy1 = MagicMock(spec=SignalProxy)
        proxy1.annotations = {"model": "efish", "channel": "A"}
        proxy2 = MagicMock(spec=SignalProxy)
        proxy2.annotations = {"model": "efish", "channel": "B"}

        strategy = BandpassFilterStrategy(low=500, high=3000)
        input_proxies = BandpassFilterStrategyProxyInput(inputs=[proxy1, proxy2])

        result = strategy.infer_execution_plan(input_proxies)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].input_proxies_dict["inputs"]), 1)
        self.assertEqual(len(result[1].input_proxies_dict["inputs"]), 1)
        self.assertEqual(
            result[0].output_annotations[0],
            {"model": "efish", "channel": "A", "filtered": True},
        )
        self.assertEqual(
            result[1].output_annotations[0],
            {"model": "efish", "channel": "B", "filtered": True},
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.processing.strategies.bandpass_filter_strategy.filtfilt")
    @patch("nexus.processing.strategies.bandpass_filter_strategy.butter")
    def test_apply(self, mock_butter: MagicMock, mock_filtfilt: MagicMock) -> None:
        """Test that apply returns AnalogSignal with same metadata as input using mocked scipy."""
        mock_butter.return_value = (np.array([1.0, 0.5]), np.array([1.0, -0.5]))
        filtered_array = np.array([[0.5], [1.5], [2.5]])
        mock_filtfilt.return_value = filtered_array

        signal = AnalogSignal(
            np.array([[1.0], [2.0], [3.0]]) * pq.mV,
            sampling_rate=10.0 * pq.kHz,
            t_start=0.5 * pq.s,
            name="ch1",
        )
        strategy = BandpassFilterStrategy(low=500, high=3000)
        input_data = BandpassFilterStrategyDataInput(inputs=[signal])

        result = strategy.apply(input_data)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AnalogSignal)
        self.assertEqual(result[0].sampling_rate, signal.sampling_rate)
        self.assertEqual(result[0].t_start, signal.t_start)
        self.assertEqual(result[0].units, signal.units)
        self.assertEqual(result[0].name, signal.name)
        np.testing.assert_array_equal(result[0].magnitude, filtered_array)
        mock_butter.assert_called_once()
        mock_filtfilt.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_apply_raises_type_error_when_input_is_not_analog_signal(self) -> None:
        """Test that apply raises TypeError when a SpikeTrain is passed instead of AnalogSignal."""
        spike_train = SpikeTrain(np.array([0.1, 0.2]) * pq.s, t_stop=1.0 * pq.s)
        strategy = BandpassFilterStrategy()
        input_data = BandpassFilterStrategyDataInput(inputs=[spike_train])

        with self.assertRaises(TypeError):
            strategy.apply(input_data)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_apply_raises_value_error_when_frequency_exceeds_nyquist(self) -> None:
        """Test that apply raises ValueError when either cutoff frequency exceeds Nyquist."""
        scenarios = [
            Scenario(
                name="low frequency exceeds Nyquist",
                given=Given(data={"low": 600, "high": 800}),
                expected=Expected(exceptions={"apply": ValueError}),
            ),
            Scenario(
                name="high frequency exceeds Nyquist",
                given=Given(data={"low": 100, "high": 600}),
                expected=Expected(exceptions={"apply": ValueError}),
            ),
        ]

        signal = AnalogSignal(
            np.array([[1.0], [2.0], [3.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,  # Nyquist = 500 Hz
        )

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                strategy = BandpassFilterStrategy(
                    low=scenario.given.data["low"],
                    high=scenario.given.data["high"],
                )
                input_data = BandpassFilterStrategyDataInput(inputs=[signal])

                with self.assertRaises(scenario.expected.exceptions["apply"]):
                    strategy.apply(input_data)


    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        strategy = BandpassFilterStrategy()

        self.assertIs(strategy.filter_criteria_type, BandpassFilterStrategyFilterCriteria)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_proxy_input_type(self) -> None:
        strategy = BandpassFilterStrategy()

        self.assertIs(strategy.proxy_input_type, BandpassFilterStrategyProxyInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        strategy = BandpassFilterStrategy()

        self.assertIs(strategy.data_input_type, BandpassFilterStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        strategy = BandpassFilterStrategy()
        filter_fields = {f.name for f in dataclasses.fields(strategy.filter_criteria_type)}
        proxy_fields = {f.name for f in dataclasses.fields(strategy.proxy_input_type)}
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, proxy_fields)
        self.assertEqual(proxy_fields, data_fields)


class TestIntegrationBandpassFilterStrategy(unittest.TestCase):
    """Integration tests for BandpassFilterStrategy with real scipy filtering."""

    def setUp(self) -> None:
        t = np.linspace(0, 1, 10000, endpoint=False)
        self.signal = AnalogSignal(
            np.sin(2 * np.pi * 100 * t) * pq.mV,
            sampling_rate=10.0 * pq.kHz,
            t_start=0.0 * pq.s,
            name="voltage",
        )

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply(self) -> None:
        """Test that apply returns AnalogSignal with same length and metadata as the input."""
        strategy = BandpassFilterStrategy(low=50, high=200)
        input_data = BandpassFilterStrategyDataInput(inputs=[self.signal])

        result = strategy.apply(input_data)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AnalogSignal)
        self.assertEqual(len(result[0]), len(self.signal))
        self.assertEqual(result[0].sampling_rate, self.signal.sampling_rate)
        self.assertEqual(result[0].t_start, self.signal.t_start)
        self.assertEqual(result[0].units, self.signal.units)
        self.assertEqual(result[0].name, self.signal.name)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply_attenuates_out_of_band_signal(self) -> None:
        """Test that the in-band signal retains power while the out-of-band signal is attenuated."""
        scenarios = [
            Scenario(
                name="50–200 Hz bandpass: in-band 100 Hz vs out-of-band 2000 Hz",
                given=Given(
                    data={
                        "in_band_freq": 100,
                        "out_of_band_freq": 2000,
                        "low": 50,
                        "high": 200,
                    }
                ),
                expected=Expected(data={"min_power_ratio": 10}),
            ),
            Scenario(
                name="1000–3000 Hz bandpass: in-band 2000 Hz vs out-of-band 100 Hz",
                given=Given(
                    data={
                        "in_band_freq": 2000,
                        "out_of_band_freq": 100,
                        "low": 1000,
                        "high": 3000,
                    }
                ),
                expected=Expected(data={"min_power_ratio": 10}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                t = np.linspace(0, 1, 10000, endpoint=False)

                in_band_signal = AnalogSignal(
                    np.sin(2 * np.pi * scenario.given.data["in_band_freq"] * t) * pq.mV,
                    sampling_rate=10.0 * pq.kHz,
                )
                out_of_band_signal = AnalogSignal(
                    np.sin(2 * np.pi * scenario.given.data["out_of_band_freq"] * t)
                    * pq.mV,
                    sampling_rate=10.0 * pq.kHz,
                )
                strategy = BandpassFilterStrategy(
                    low=scenario.given.data["low"],
                    high=scenario.given.data["high"],
                )

                result_in = strategy.apply(
                    BandpassFilterStrategyDataInput(inputs=[in_band_signal])
                )
                result_out = strategy.apply(
                    BandpassFilterStrategyDataInput(inputs=[out_of_band_signal])
                )

                power_in = float(np.var(result_in[0].magnitude))
                power_out = float(np.var(result_out[0].magnitude))

                self.assertGreater(
                    power_in, power_out * scenario.expected.data["min_power_ratio"]
                )
