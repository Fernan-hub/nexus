"""Tests for SpikeExtractionStrategy."""

import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal, SpikeTrain

from nexus.core.interfaces import SignalProxy
from nexus.processing.strategies.spike_extraction_strategy import (
    SpikeExtractionStrategy,
    SpikeExtractionStrategyDataInput,
    SpikeExtractionStrategyFilterCriteria,
    SpikeExtractionStrategyProxyInput,
)
from tests.utils import Expected, Given, Scenario


class TestSpikeExtractionStrategy(unittest.TestCase):
    """Unit tests for SpikeExtractionStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_infer_execution_plan(self) -> None:
        """Test that each proxy gets its own NodeDefinition with its annotations plus spike=True."""
        proxy1 = MagicMock(spec=SignalProxy)
        proxy1.annotations = {"model": "efish", "channel": "A"}
        proxy2 = MagicMock(spec=SignalProxy)
        proxy2.annotations = {"model": "efish", "channel": "B"}

        strategy = SpikeExtractionStrategy(threshold=0.0 * pq.mV)
        input_proxies = SpikeExtractionStrategyProxyInput(inputs=[proxy1, proxy2])

        result = strategy.infer_execution_plan(input_proxies)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].input_proxies_dict["inputs"]), 1)
        self.assertEqual(len(result[1].input_proxies_dict["inputs"]), 1)
        self.assertEqual(
            result[0].output_annotations[0],
            {"model": "efish", "channel": "A", "spike": True},
        )
        self.assertEqual(
            result[1].output_annotations[0],
            {"model": "efish", "channel": "B", "spike": True},
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.processing.strategies.spike_extraction_strategy.threshold_detection")
    def test_apply(self, mock_threshold_detection: MagicMock) -> None:
        """Test that apply returns the SpikeTrain from threshold_detection with name set from input."""
        expected_spike_train = SpikeTrain(
            np.array([0.1, 0.3]) * pq.s,
            t_stop=1.0 * pq.s,
        )
        mock_threshold_detection.return_value = expected_spike_train

        signal = AnalogSignal(
            np.array([[0.0], [5.0], [0.0], [5.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
            name="ch1",
        )
        strategy = SpikeExtractionStrategy(threshold=2.5 * pq.mV)
        input_data = SpikeExtractionStrategyDataInput(inputs=[signal])

        result = strategy.apply(input_data)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], SpikeTrain)
        self.assertEqual(result[0].name, "ch1")
        mock_threshold_detection.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_apply_raises_type_error_when_input_is_not_analog_signal(self) -> None:
        """Test that apply raises TypeError when a SpikeTrain is passed instead of AnalogSignal."""
        spike_train = SpikeTrain(np.array([0.1]) * pq.s, t_stop=1.0 * pq.s)
        strategy = SpikeExtractionStrategy()
        input_data = SpikeExtractionStrategyDataInput(inputs=[spike_train])

        with self.assertRaises(TypeError):
            strategy.apply(input_data)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        """Returns SpikeExtractionStrategyFilterCriteria as filter_criteria_type."""
        strategy = SpikeExtractionStrategy()

        self.assertIs(
            strategy.filter_criteria_type, SpikeExtractionStrategyFilterCriteria
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_proxy_input_type(self) -> None:
        """Returns SpikeExtractionStrategyProxyInput as proxy_input_type."""
        strategy = SpikeExtractionStrategy()

        self.assertIs(strategy.proxy_input_type, SpikeExtractionStrategyProxyInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        """Returns SpikeExtractionStrategyDataInput as data_input_type."""
        strategy = SpikeExtractionStrategy()

        self.assertIs(strategy.data_input_type, SpikeExtractionStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        """All three input type properties share identical field names."""
        strategy = SpikeExtractionStrategy()
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        proxy_fields = {f.name for f in dataclasses.fields(strategy.proxy_input_type)}
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, proxy_fields)
        self.assertEqual(proxy_fields, data_fields)


class TestIntegrationSpikeExtractionStrategy(unittest.TestCase):
    """Integration tests for SpikeExtractionStrategy with real elephant threshold detection."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply(self) -> None:
        """Test that apply returns a SpikeTrain whose spike count matches crossings of the threshold."""
        scenarios = [
            Scenario(
                name="signal with two upward crossings above threshold",
                given=Given(
                    data={
                        "values": [
                            [0.0],
                            [0.0],
                            [5.0],
                            [5.0],
                            [0.0],
                            [0.0],
                            [5.0],
                            [5.0],
                            [0.0],
                            [0.0],
                        ],
                        "threshold": 2.5 * pq.mV,
                        "above_threshold": True,
                        "name": "voltage",
                    }
                ),
                expected=Expected(data={"spike_count": 2, "name": "voltage"}),
            ),
            Scenario(
                name="signal that never exceeds threshold",
                given=Given(
                    data={
                        "values": [[0.0], [1.0], [0.5], [0.8], [0.3]],
                        "threshold": 5.0 * pq.mV,
                        "above_threshold": True,
                        "name": "silent",
                    }
                ),
                expected=Expected(data={"spike_count": 0, "name": "silent"}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                d = scenario.given.data
                e = scenario.expected.data

                signal = AnalogSignal(
                    np.array(d["values"]) * pq.mV,
                    sampling_rate=1.0 * pq.kHz,
                    name=d["name"],
                )
                strategy = SpikeExtractionStrategy(
                    threshold=d["threshold"],
                    above_threshold=d["above_threshold"],
                )
                input_data = SpikeExtractionStrategyDataInput(inputs=[signal])

                result = strategy.apply(input_data)

                self.assertEqual(len(result), 1)
                self.assertIsInstance(result[0], SpikeTrain)
                self.assertEqual(len(result[0]), e["spike_count"])
                self.assertEqual(result[0].name, e["name"])
