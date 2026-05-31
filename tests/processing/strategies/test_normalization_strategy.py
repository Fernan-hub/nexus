"""Tests for NormalizationStrategy."""

import dataclasses
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.core.interfaces import SignalProxy
from nexus.processing.strategies.normalization_strategy import (
    NormalizationStrategy,
    NormalizationStrategyDataInput,
    NormalizationStrategyFilterCriteria,
    NormalizationStrategyProxyInput,
)
from tests.utils import Expected, Given, Scenario


class TestNormalizationStrategy(unittest.TestCase):
    """Unit tests for NormalizationStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_infer_execution_plan(self) -> None:
        """Test that each proxy gets its own NodeDefinition with its annotations plus normalized=True."""
        proxy1 = MagicMock(spec=SignalProxy)
        proxy1.annotations = {"model": "efish", "channel": "A"}
        proxy2 = MagicMock(spec=SignalProxy)
        proxy2.annotations = {"type": "voltage"}

        strategy = NormalizationStrategy()
        input_proxies = NormalizationStrategyProxyInput(inputs=[proxy1, proxy2])

        result = strategy.infer_execution_plan(input_proxies)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].input_proxies_dict["inputs"]), 1)
        self.assertEqual(len(result[1].input_proxies_dict["inputs"]), 1)
        self.assertEqual(
            result[0].output_annotations[0],
            {"model": "efish", "channel": "A", "normalized": True},
        )
        self.assertEqual(
            result[1].output_annotations[0], {"type": "voltage", "normalized": True}
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        """Returns NormalizationStrategyFilterCriteria as filter_criteria_type."""
        strategy = NormalizationStrategy()

        self.assertIs(
            strategy.filter_criteria_type, NormalizationStrategyFilterCriteria
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_proxy_input_type(self) -> None:
        """Returns NormalizationStrategyProxyInput as proxy_input_type."""
        strategy = NormalizationStrategy()

        self.assertIs(strategy.proxy_input_type, NormalizationStrategyProxyInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        """Returns NormalizationStrategyDataInput as data_input_type."""
        strategy = NormalizationStrategy()

        self.assertIs(strategy.data_input_type, NormalizationStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        """All three input type properties share identical field names."""
        strategy = NormalizationStrategy()
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        proxy_fields = {f.name for f in dataclasses.fields(strategy.proxy_input_type)}
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, proxy_fields)
        self.assertEqual(proxy_fields, data_fields)


class TestIntegrationNormalizationStrategy(unittest.TestCase):
    """Integration tests for NormalizationStrategy."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply(self) -> None:
        """Test that apply scales signal values so the minimum equals low and maximum equals high."""
        scenarios = [
            Scenario(
                name="default range [0, 1] mV",
                given=Given(
                    data={
                        "values": [1.0, 2.0, 3.0, 4.0],
                        "low": 0 * pq.mV,
                        "high": 1 * pq.mV,
                    }
                ),
                expected=Expected(
                    data={
                        "values": [0.0, 1 / 3, 2 / 3, 1.0],
                        "units": pq.mV,
                    }
                ),
            ),
            Scenario(
                name="custom range [-1, 1] mV",
                given=Given(
                    data={
                        "values": [0.0, 1.0, 2.0],
                        "low": -1 * pq.mV,
                        "high": 1 * pq.mV,
                    }
                ),
                expected=Expected(
                    data={
                        "values": [-1.0, 0.0, 1.0],
                        "units": pq.mV,
                    }
                ),
            ),
            Scenario(
                name="range [2, 5] mV",
                given=Given(
                    data={
                        "values": [1.0, 2.0, 3.0, 4.0],
                        "low": 2 * pq.mV,
                        "high": 5 * pq.mV,
                    }
                ),
                expected=Expected(
                    data={
                        "values": [2.0, 3.0, 4.0, 5.0],
                        "units": pq.mV,
                    }
                ),
            ),
            Scenario(
                name="constant signal maps entirely to low bound",
                given=Given(
                    data={
                        "values": [3.0, 3.0, 3.0],
                        "low": 0 * pq.mV,
                        "high": 1 * pq.mV,
                    }
                ),
                expected=Expected(
                    data={
                        "values": [0.0, 0.0, 0.0],
                        "units": pq.mV,
                    }
                ),
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
                strategy = NormalizationStrategy(low=d["low"], high=d["high"])
                input_data = NormalizationStrategyDataInput(inputs=[signal])

                result = strategy.apply(input_data)

                self.assertEqual(len(result), 1)
                self.assertIsInstance(result[0], AnalogSignal)
                self.assertEqual(result[0].units, e["units"])
                self.assertEqual(len(result[0]), len(signal))
                np.testing.assert_array_almost_equal(
                    result[0].magnitude.flatten(), e["values"]
                )

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply_preserves_metadata(self) -> None:
        """Test that apply preserves sampling rate, t_start, name, and output units equal high.units."""
        signal = AnalogSignal(
            np.array([1.0, 2.0, 3.0]) * pq.mV,
            sampling_rate=30.0 * pq.kHz,
            t_start=0.5 * pq.s,
            name="ch1",
        )
        strategy = NormalizationStrategy(low=0 * pq.mV, high=1 * pq.mV)
        input_data = NormalizationStrategyDataInput(inputs=[signal])

        result = strategy.apply(input_data)

        self.assertEqual(result[0].sampling_rate, signal.sampling_rate)
        self.assertEqual(result[0].t_start, signal.t_start)
        self.assertEqual(result[0].units, pq.mV)
        self.assertEqual(result[0].name, signal.name)
