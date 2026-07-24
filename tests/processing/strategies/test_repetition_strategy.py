"""Tests for RepetitionStrategy."""

import dataclasses
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.core.interfaces import SignalProxy
from nexus.processing.strategies.repetition_strategy import (
    RepetitionStrategy,
    RepetitionStrategyDataInput,
    RepetitionStrategyFilterCriteria,
    RepetitionStrategyProxyInput,
)
from tests.utils import Expected, Given, Scenario


class TestRepetitionStrategy(unittest.TestCase):
    """Unit tests for RepetitionStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_infer_execution_plan(self) -> None:
        """Test each proxy gets its own NodeDefinition with repeated=True added."""
        proxy1 = MagicMock(spec=SignalProxy)
        proxy1.annotations = {"model": "efish", "channel": "A"}
        proxy2 = MagicMock(spec=SignalProxy)
        proxy2.annotations = {"type": "voltage"}

        strategy = RepetitionStrategy(repetitions=3)
        input_proxies = RepetitionStrategyProxyInput(inputs=[proxy1, proxy2])

        result = strategy.infer_execution_plan(input_proxies)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].input_proxies_dict["inputs"]), 1)
        self.assertEqual(len(result[1].input_proxies_dict["inputs"]), 1)
        self.assertEqual(
            result[0].output_annotations[0],
            {"model": "efish", "channel": "A", "repeated": True},
        )
        self.assertEqual(
            result[1].output_annotations[0], {"type": "voltage", "repeated": True}
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_init_raises_value_error_when_repetitions_is_invalid(self) -> None:
        """Test init raises ValueError for a non-positive repetition count."""
        scenarios = [
            Scenario(
                name="zero repetitions",
                given=Given(data={"repetitions": 0}),
                expected=Expected(exceptions={"init": ValueError}),
            ),
            Scenario(
                name="negative repetitions",
                given=Given(data={"repetitions": -5}),
                expected=Expected(exceptions={"init": ValueError}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                with self.assertRaises(scenario.expected.exceptions["init"]):
                    RepetitionStrategy(repetitions=scenario.given.data["repetitions"])

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        """Returns RepetitionStrategyFilterCriteria as filter_criteria_type."""
        strategy = RepetitionStrategy(repetitions=1)

        self.assertIs(strategy.filter_criteria_type, RepetitionStrategyFilterCriteria)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_proxy_input_type(self) -> None:
        """Returns RepetitionStrategyProxyInput as proxy_input_type."""
        strategy = RepetitionStrategy(repetitions=1)

        self.assertIs(strategy.proxy_input_type, RepetitionStrategyProxyInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        """Returns RepetitionStrategyDataInput as data_input_type."""
        strategy = RepetitionStrategy(repetitions=1)

        self.assertIs(strategy.data_input_type, RepetitionStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        """All three input type properties share identical field names."""
        strategy = RepetitionStrategy(repetitions=1)
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        proxy_fields = {f.name for f in dataclasses.fields(strategy.proxy_input_type)}
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, proxy_fields)
        self.assertEqual(proxy_fields, data_fields)


class TestIntegrationRepetitionStrategy(unittest.TestCase):
    """Integration tests for RepetitionStrategy."""

    def setUp(self) -> None:
        self.signal = AnalogSignal(
            np.array([[1.0], [2.0], [3.0], [4.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
            t_start=0.0 * pq.s,
            name="ch1",
        )

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply(self) -> None:
        """Test apply repeats input cyclically N times over N x duration."""
        scenarios = [
            Scenario(
                name="2 repetitions",
                given=Given(data={"repetitions": 2}),
                expected=Expected(
                    data={
                        "length": 8,
                        "duration_factor": 2,
                        "values": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                    }
                ),
            ),
            Scenario(
                name="3 repetitions",
                given=Given(data={"repetitions": 3}),
                expected=Expected(
                    data={
                        "length": 12,
                        "duration_factor": 3,
                        "values": [
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                        ],
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                e = scenario.expected.data
                strategy = RepetitionStrategy(
                    repetitions=scenario.given.data["repetitions"]
                )
                input_data = RepetitionStrategyDataInput(inputs=[self.signal])

                result = strategy.apply(input_data)

                self.assertEqual(len(result), 1)
                self.assertIsInstance(result[0], AnalogSignal)
                self.assertEqual(len(result[0]), e["length"])
                self.assertEqual(result[0].units, self.signal.units)
                self.assertAlmostEqual(
                    float(result[0].duration.rescale(pq.s)),
                    e["duration_factor"] * float(self.signal.duration.rescale(pq.s)),
                    places=9,
                )
                np.testing.assert_array_almost_equal(
                    result[0].magnitude.flatten(), e["values"]
                )
