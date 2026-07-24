"""Tests for StandardizationStrategy."""

import dataclasses
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.core.interfaces import SignalProxy
from nexus.processing.strategies.standardization_strategy import (
    StandardizationStrategy,
    StandardizationStrategyDataInput,
    StandardizationStrategyFilterCriteria,
    StandardizationStrategyProxyInput,
)


class TestStandardizationStrategy(unittest.TestCase):
    """Unit tests for StandardizationStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_infer_execution_plan(self) -> None:
        """Test each proxy gets its own NodeDefinition with standardized=True added."""
        proxy1 = MagicMock(spec=SignalProxy)
        proxy1.annotations = {"model": "efish", "channel": "A"}
        proxy2 = MagicMock(spec=SignalProxy)
        proxy2.annotations = {"type": "voltage"}

        strategy = StandardizationStrategy()
        input_proxies = StandardizationStrategyProxyInput(inputs=[proxy1, proxy2])

        result = strategy.infer_execution_plan(input_proxies)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].input_proxies_dict["inputs"]), 1)
        self.assertEqual(len(result[1].input_proxies_dict["inputs"]), 1)
        self.assertEqual(
            result[0].output_annotations[0],
            {"model": "efish", "channel": "A", "standardized": True},
        )
        self.assertEqual(
            result[1].output_annotations[0], {"type": "voltage", "standardized": True}
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        """Returns StandardizationStrategyFilterCriteria as filter_criteria_type."""
        strategy = StandardizationStrategy()

        self.assertIs(
            strategy.filter_criteria_type, StandardizationStrategyFilterCriteria
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_proxy_input_type(self) -> None:
        """Returns StandardizationStrategyProxyInput as proxy_input_type."""
        strategy = StandardizationStrategy()

        self.assertIs(strategy.proxy_input_type, StandardizationStrategyProxyInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        """Returns StandardizationStrategyDataInput as data_input_type."""
        strategy = StandardizationStrategy()

        self.assertIs(strategy.data_input_type, StandardizationStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        """All three input type properties share identical field names."""
        strategy = StandardizationStrategy()
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        proxy_fields = {f.name for f in dataclasses.fields(strategy.proxy_input_type)}
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, proxy_fields)
        self.assertEqual(proxy_fields, data_fields)


class TestIntegrationStandardizationStrategy(unittest.TestCase):
    """Integration tests for StandardizationStrategy."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply(self) -> None:
        """Test apply produces a signal with zero mean and unit standard deviation."""
        signal = AnalogSignal(
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]) * pq.mV,
            sampling_rate=1.0 * pq.kHz,
        )
        strategy = StandardizationStrategy()
        input_data = StandardizationStrategyDataInput(inputs=[signal])

        result = strategy.apply(input_data)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), len(signal))
        self.assertAlmostEqual(float(np.mean(result[0].magnitude)), 0.0, places=10)
        self.assertAlmostEqual(float(np.std(result[0].magnitude)), 1.0, places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_apply_preserves_units_sampling_rate_and_shape(self) -> None:
        """Test that apply does not alter the signal units, sampling rate, or shape."""
        signal = AnalogSignal(
            np.array([1.0, 3.0, 5.0, 7.0]) * pq.mV,
            sampling_rate=30.0 * pq.kHz,
            t_start=0.1 * pq.s,
        )
        strategy = StandardizationStrategy()
        input_data = StandardizationStrategyDataInput(inputs=[signal])

        result = strategy.apply(input_data)

        self.assertEqual(result[0].units, signal.units)
        self.assertEqual(result[0].sampling_rate, signal.sampling_rate)
        self.assertEqual(result[0].t_start, signal.t_start)
        self.assertEqual(result[0].shape, signal.shape)
