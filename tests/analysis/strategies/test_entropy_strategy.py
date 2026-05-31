"""Tests for EntropyStrategy."""

import math
import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.analysis.estimators.entropy import DiscreteEntropyConfig, KernelEntropyConfig
from nexus.analysis.results.vector_result import VectorResult
from nexus.analysis.strategies.entropy_strategy import (
    EntropyStrategy,
    EntropyStrategyDataInput,
    EntropyStrategyFilterCriteria,
)
from tests.utils import Expected, Given, Scenario


def _make_signal(name: str | None = None) -> AnalogSignal:
    return AnalogSignal(
        np.array([1.0, 2.0, 3.0]) * pq.mV,
        sampling_rate=1.0 * pq.kHz,
        name=name,
    )


class TestEntropyStrategy(unittest.TestCase):
    """Unit tests for EntropyStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis(self, mock_estimator_class: MagicMock) -> None:
        """Test that run_analysis calls the estimator once per signal and returns a VectorResult."""
        scenarios = [
            Scenario(
                name="single signal",
                given=Given(data={"signals": [_make_signal()]}),
                expected=Expected(data={"row_count": 1}),
            ),
            Scenario(
                name="multiple signals",
                given=Given(
                    data={"signals": [_make_signal(), _make_signal(), _make_signal()]}
                ),
                expected=Expected(data={"row_count": 3}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = 0.5

                strategy = EntropyStrategy(DiscreteEntropyConfig())
                data_input = EntropyStrategyDataInput(
                    data=scenario.given.data["signals"]
                )

                result = strategy.run_analysis(data_input)

                self.assertIsInstance(result, VectorResult)
                self.assertEqual(result.algorithm, "entropy")
                self.assertEqual(
                    len(result.vector), scenario.expected.data["row_count"]
                )
                self.assertEqual(list(result.vector.columns), ["entropy"])
                self.assertEqual(
                    mock_estimator_class.call_count, scenario.expected.data["row_count"]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis_result_values_come_from_estimator(
        self, mock_estimator_class: MagicMock
    ) -> None:
        """Test that the values in the result vector come from the estimator."""
        mock_estimator_class.return_value.result.side_effect = [0.42, 1.1]

        strategy = EntropyStrategy(DiscreteEntropyConfig())
        data_input = EntropyStrategyDataInput(data=[_make_signal(), _make_signal()])

        result = strategy.run_analysis(data_input)

        np.testing.assert_array_almost_equal(
            result.vector["entropy"].tolist(), [0.42, 1.1]
        )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis_cond_column_name(
        self, mock_estimator_class: MagicMock
    ) -> None:
        """Test that the cond column is named after the cond signal, falling back to a default."""
        scenarios = [
            Scenario(
                name="named cond signal",
                given=Given(data={"cond": _make_signal(name="reference")}),
                expected=Expected(data={"column": "cond_entropy_given_reference"}),
            ),
            Scenario(
                name="unnamed cond signal",
                given=Given(data={"cond": _make_signal(name=None)}),
                expected=Expected(data={"column": "cond_entropy"}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = 0.5

                strategy = EntropyStrategy(DiscreteEntropyConfig())
                data_input = EntropyStrategyDataInput(
                    data=[_make_signal()],
                    cond=[scenario.given.data["cond"]],
                )

                result = strategy.run_analysis(data_input)

                self.assertEqual(
                    list(result.vector.columns), [scenario.expected.data["column"]]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        strategy = EntropyStrategy(DiscreteEntropyConfig())

        self.assertIs(strategy.filter_criteria_type, EntropyStrategyFilterCriteria)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        strategy = EntropyStrategy(DiscreteEntropyConfig())

        self.assertIs(strategy.data_input_type, EntropyStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        strategy = EntropyStrategy(DiscreteEntropyConfig())
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, data_fields)


class TestIntegrationEntropyStrategy(unittest.TestCase):
    """Integration tests for EntropyStrategy against real infomeasure estimators."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_entropy_of_constant_signal_is_zero(self) -> None:
        """A signal with a single unique value has zero entropy."""
        sig = AnalogSignal(np.zeros(100) * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = EntropyStrategy(DiscreteEntropyConfig()).run_analysis(
            EntropyStrategyDataInput(data=[sig])
        )

        self.assertAlmostEqual(result.vector["entropy"][0], 0.0, places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_entropy_of_uniform_binary_signal(self) -> None:
        """A perfectly uniform binary signal has entropy = ln(2) nats (infomeasure default base)."""
        data = np.tile([0.0, 1.0], 50)
        sig = AnalogSignal(data * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = EntropyStrategy(DiscreteEntropyConfig()).run_analysis(
            EntropyStrategyDataInput(data=[sig])
        )

        self.assertAlmostEqual(result.vector["entropy"][0], np.log(2), places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_conditional_entropy_of_signal_given_itself_is_zero(self) -> None:
        """H(X | X) = 0 because X is fully determined given itself."""
        data = np.tile([0.0, 1.0], 50)
        sig = AnalogSignal(data * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = EntropyStrategy(DiscreteEntropyConfig()).run_analysis(
            EntropyStrategyDataInput(data=[sig], cond=[sig])
        )

        self.assertAlmostEqual(result.vector["cond_entropy"][0], 0.0, places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    @pytest.mark.slow
    def test_kernel_entropy_produces_finite_result(self) -> None:
        """KernelEntropyConfig wires up correctly: the strategy runs without error and returns a finite value."""
        rng = np.random.default_rng(0)
        sig = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = EntropyStrategy(KernelEntropyConfig()).run_analysis(
            EntropyStrategyDataInput(data=[sig])
        )

        self.assertIsInstance(result, VectorResult)
        self.assertTrue(math.isfinite(result.vector["entropy"][0]))

    @pytest.mark.integration
    @pytest.mark.strategy
    @pytest.mark.slow
    def test_kernel_conditional_entropy_produces_finite_result(self) -> None:
        """KernelEntropyConfig conditional path wires up correctly and returns a finite value."""
        rng = np.random.default_rng(1)
        sig = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)
        cond = AnalogSignal(rng.normal(0, 1, 100) * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = EntropyStrategy(KernelEntropyConfig()).run_analysis(
            EntropyStrategyDataInput(data=[sig], cond=[cond])
        )

        self.assertIsInstance(result, VectorResult)
        self.assertTrue(math.isfinite(result.vector["cond_entropy"][0]))
