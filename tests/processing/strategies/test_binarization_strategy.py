"""Tests for BinarizationStrategy."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal, SpikeTrain

from nexus.core.interfaces import SignalProxy
from nexus.processing.strategies.binarization_strategy import (
    BinarizationStrategy,
    BinarizationStrategyDataInput,
    BinarizationStrategyProxyInput,
)
from tests.utils import Expected, Given, Scenario


class TestBinarizationStrategy(unittest.TestCase):
    """Unit tests for BinarizationStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_infer_execution_plan(self) -> None:
        """Test that each proxy gets its own NodeDefinition with its annotations plus binary=True."""
        proxy1 = MagicMock(spec=SignalProxy)
        proxy1.annotations = {"model": "efish", "channel": "A"}
        proxy2 = MagicMock(spec=SignalProxy)
        proxy2.annotations = {"type": "voltage"}

        strategy = BinarizationStrategy(threshold=0.0 * pq.mV)
        input_proxies = BinarizationStrategyProxyInput(inputs=[proxy1, proxy2])

        result = strategy.infer_execution_plan(input_proxies)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].input_proxies_dict["inputs"]), 1)
        self.assertEqual(len(result[1].input_proxies_dict["inputs"]), 1)
        self.assertEqual(
            result[0].output_annotations[0],
            {"model": "efish", "channel": "A", "binary": True},
        )
        self.assertEqual(
            result[1].output_annotations[0], {"type": "voltage", "binary": True}
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_apply_raises_type_error_when_input_is_not_analog_signal(self) -> None:
        """Test that apply raises TypeError when a SpikeTrain is passed instead of AnalogSignal."""
        spike_train = SpikeTrain(np.array([0.1]) * pq.s, t_stop=1.0 * pq.s)
        strategy = BinarizationStrategy()
        input_data = BinarizationStrategyDataInput(inputs=[spike_train])

        with self.assertRaises(TypeError):
            strategy.apply(input_data)


class TestIntegrationBinarizationStrategy(unittest.TestCase):
    """Integration tests for BinarizationStrategy."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply(self) -> None:
        """Test that apply maps values above threshold to 1 and values at or below to 0."""
        scenarios = [
            Scenario(
                name="threshold at 0 mV",
                given=Given(
                    data={
                        "values": [-1.0, 0.0, 1.0, 2.0],
                        "threshold": 0.0 * pq.mV,
                    }
                ),
                expected=Expected(data={"binary": [0.0, 0.0, 1.0, 1.0]}),
            ),
            Scenario(
                name="threshold at 2.5 mV",
                given=Given(
                    data={
                        "values": [0.0, 1.0, 2.5, 3.0, 5.0],
                        "threshold": 2.5 * pq.mV,
                    }
                ),
                expected=Expected(data={"binary": [0.0, 0.0, 0.0, 1.0, 1.0]}),
            ),
            Scenario(
                name="all values above threshold",
                given=Given(
                    data={
                        "values": [3.0, 4.0, 5.0],
                        "threshold": 2.0 * pq.mV,
                    }
                ),
                expected=Expected(data={"binary": [1.0, 1.0, 1.0]}),
            ),
            Scenario(
                name="no values above threshold",
                given=Given(
                    data={
                        "values": [0.0, 0.5, 1.0],
                        "threshold": 2.0 * pq.mV,
                    }
                ),
                expected=Expected(data={"binary": [0.0, 0.0, 0.0]}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                d = scenario.given.data
                e = scenario.expected.data

                signal = AnalogSignal(
                    np.array(d["values"]) * pq.mV,
                    sampling_rate=1.0 * pq.kHz,
                )
                strategy = BinarizationStrategy(threshold=d["threshold"])
                input_data = BinarizationStrategyDataInput(inputs=[signal])

                result = strategy.apply(input_data)

                self.assertEqual(len(result), 1)
                self.assertIsInstance(result[0], AnalogSignal)
                self.assertEqual(result[0].units, pq.dimensionless)
                self.assertEqual(len(result[0]), len(signal))
                np.testing.assert_array_equal(
                    result[0].magnitude.flatten(), e["binary"]
                )

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply_preserves_metadata(self) -> None:
        """Test that apply preserves sampling rate, t_start, name, and outputs dimensionless units."""
        signal = AnalogSignal(
            np.array([0.0, 1.0, 2.0]) * pq.mV,
            sampling_rate=30.0 * pq.kHz,
            t_start=0.5 * pq.s,
            name="ch1",
        )
        strategy = BinarizationStrategy(threshold=0.5 * pq.mV)
        input_data = BinarizationStrategyDataInput(inputs=[signal])

        result = strategy.apply(input_data)

        self.assertEqual(result[0].sampling_rate, signal.sampling_rate)
        self.assertEqual(result[0].t_start, signal.t_start)
        self.assertEqual(result[0].name, signal.name)
        self.assertEqual(result[0].units, pq.dimensionless)
