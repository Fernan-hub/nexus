"""Tests for ConcatenationStrategy."""

import dataclasses
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.core.interfaces import SignalProxy
from nexus.processing.strategies.concatenation_strategy import (
    ConcatenationStrategy,
    ConcatenationStrategyDataInput,
    ConcatenationStrategyFilterCriteria,
    ConcatenationStrategyProxyInput,
)
from tests.utils import Expected, Given, Scenario


class TestConcatenationStrategy(unittest.TestCase):
    """Unit tests for ConcatenationStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_infer_execution_plan(self) -> None:
        """Test that all proxies map to a single NodeDefinition and only common annotations are retained."""
        scenarios = [
            Scenario(
                name="partial common annotations: only shared keys retained",
                given=Given(
                    data={
                        "proxy1_annotations": {"model": "efish", "channel": "A"},
                        "proxy2_annotations": {"model": "efish", "channel": "B"},
                    }
                ),
                expected=Expected(
                    data={
                        "output_annotations": {"model": "efish", "concatenated": True},
                    }
                ),
            ),
            Scenario(
                name="all annotations shared: all keys retained",
                given=Given(
                    data={
                        "proxy1_annotations": {"model": "efish", "type": "voltage"},
                        "proxy2_annotations": {"model": "efish", "type": "voltage"},
                    }
                ),
                expected=Expected(
                    data={
                        "output_annotations": {
                            "model": "efish",
                            "type": "voltage",
                            "concatenated": True,
                        },
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                proxy1 = MagicMock(spec=SignalProxy)
                proxy1.annotations = scenario.given.data["proxy1_annotations"]
                proxy2 = MagicMock(spec=SignalProxy)
                proxy2.annotations = scenario.given.data["proxy2_annotations"]

                strategy = ConcatenationStrategy()
                input_proxies = ConcatenationStrategyProxyInput(inputs=[proxy1, proxy2])

                result = strategy.infer_execution_plan(input_proxies)

                self.assertEqual(len(result), 1)
                self.assertEqual(len(result[0].input_proxies_dict["inputs"]), 2)
                self.assertEqual(
                    result[0].output_annotations[0],
                    scenario.expected.data["output_annotations"],
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        strategy = ConcatenationStrategy()

        self.assertIs(
            strategy.filter_criteria_type, ConcatenationStrategyFilterCriteria
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_proxy_input_type(self) -> None:
        strategy = ConcatenationStrategy()

        self.assertIs(strategy.proxy_input_type, ConcatenationStrategyProxyInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        strategy = ConcatenationStrategy()

        self.assertIs(strategy.data_input_type, ConcatenationStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        strategy = ConcatenationStrategy()
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        proxy_fields = {f.name for f in dataclasses.fields(strategy.proxy_input_type)}
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, proxy_fields)
        self.assertEqual(proxy_fields, data_fields)


class TestIntegrationConcatenationStrategy(unittest.TestCase):
    """Integration tests for ConcatenationStrategy."""

    def setUp(self) -> None:
        self.signal1 = AnalogSignal(
            np.array([[1.0], [2.0], [3.0], [4.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
            t_start=0.0 * pq.s,
        )
        # t_stop of signal1 = 0 + 4/1000 = 4 ms
        self.signal2 = AnalogSignal(
            np.array([[5.0], [6.0], [7.0], [8.0]]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
            t_start=4.0 * pq.ms,
        )

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply(self) -> None:
        """Test that apply concatenates all signals into one with the summed number of samples."""
        strategy = ConcatenationStrategy()
        input_data = ConcatenationStrategyDataInput(inputs=[self.signal1, self.signal2])

        result = strategy.apply(input_data)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AnalogSignal)
        self.assertEqual(len(result[0]), len(self.signal1) + len(self.signal2))
        self.assertEqual(result[0].units, self.signal1.units)
        np.testing.assert_array_almost_equal(
            result[0].magnitude.flatten(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply_sorts_by_t_start_before_concatenating(self) -> None:
        """Test that apply correctly orders signals by t_start regardless of input order."""
        strategy = ConcatenationStrategy()
        input_data = ConcatenationStrategyDataInput(inputs=[self.signal2, self.signal1])

        result = strategy.apply(input_data)

        np.testing.assert_array_almost_equal(
            result[0].magnitude.flatten(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
