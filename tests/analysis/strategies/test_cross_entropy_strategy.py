"""Tests for CrossEntropyStrategy."""

import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import quantities as pq
from neo.core import AnalogSignal

from nexus.analysis.estimators.entropy import DiscreteEntropyConfig
from nexus.analysis.results.matrix_result import MatrixResult
from nexus.analysis.strategies.cross_entropy_strategy import (
    CrossEntropyStrategy,
    CrossEntropyStrategyDataInput,
    CrossEntropyStrategyFilterCriteria,
)
from tests.utils import Expected, Given, Scenario


def _make_signal(name: str | None = None) -> AnalogSignal:
    return AnalogSignal(
        np.array([1.0, 2.0, 3.0]) * pq.mV,
        sampling_rate=1.0 * pq.kHz,
        name=name,
    )


class TestCrossEntropyStrategy(unittest.TestCase):
    """Unit tests for CrossEntropyStrategy."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis(self, mock_estimator_class: MagicMock) -> None:
        """Test that run_analysis computes one cross-entropy value per (p, q) pair."""
        scenarios = [
            Scenario(
                name="one p one q",
                given=Given(
                    data={
                        "data_p": [_make_signal()],
                        "data_q": [_make_signal()],
                    }
                ),
                expected=Expected(data={"element_count": 1}),
            ),
            Scenario(
                name="two p two q",
                given=Given(
                    data={
                        "data_p": [_make_signal(), _make_signal()],
                        "data_q": [_make_signal(), _make_signal()],
                    }
                ),
                expected=Expected(data={"element_count": 4}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = 0.5

                strategy = CrossEntropyStrategy(DiscreteEntropyConfig())
                data_input = CrossEntropyStrategyDataInput(
                    data_p=scenario.given.data["data_p"],
                    data_q=scenario.given.data["data_q"],
                )

                result = strategy.run_analysis(data_input)

                self.assertIsInstance(result, MatrixResult)
                self.assertEqual(result.algorithm, "cross_entropy")
                self.assertEqual(
                    len(result.matrix), scenario.expected.data["element_count"]
                )
                self.assertEqual(
                    mock_estimator_class.call_count,
                    scenario.expected.data["element_count"],
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.analysis.estimators.entropy.DiscreteEntropyEstimator")
    def test_run_analysis_labels(self, mock_estimator_class: MagicMock) -> None:
        """Test that signal names are used as labels or indices when absent."""
        scenarios = [
            Scenario(
                name="named signals",
                given=Given(
                    data={
                        "data_p": [_make_signal(name="dist_p")],
                        "data_q": [_make_signal(name="dist_q")],
                    }
                ),
                expected=Expected(data={"row": "dist_p", "column": "dist_q"}),
            ),
            Scenario(
                name="unnamed signals",
                given=Given(
                    data={
                        "data_p": [_make_signal(name=None)],
                        "data_q": [_make_signal(name=None)],
                    }
                ),
                expected=Expected(data={"row": "data_p_0", "column": "data_q_0"}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_estimator_class.reset_mock()
                mock_estimator_class.return_value.result.return_value = 0.5

                strategy = CrossEntropyStrategy(DiscreteEntropyConfig())
                data_input = CrossEntropyStrategyDataInput(
                    data_p=scenario.given.data["data_p"],
                    data_q=scenario.given.data["data_q"],
                )

                result = strategy.run_analysis(data_input)

                self.assertEqual(result.matrix[0].row, scenario.expected.data["row"])
                self.assertEqual(
                    result.matrix[0].column, scenario.expected.data["column"]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_filter_criteria_type(self) -> None:
        """Returns CrossEntropyStrategyFilterCriteria as filter_criteria_type."""
        strategy = CrossEntropyStrategy(DiscreteEntropyConfig())

        self.assertIs(strategy.filter_criteria_type, CrossEntropyStrategyFilterCriteria)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_data_input_type(self) -> None:
        """Returns CrossEntropyStrategyDataInput as data_input_type."""
        strategy = CrossEntropyStrategy(DiscreteEntropyConfig())

        self.assertIs(strategy.data_input_type, CrossEntropyStrategyDataInput)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_input_types_share_field_names(self) -> None:
        """filter_criteria_type and data_input_type share identical field names."""
        strategy = CrossEntropyStrategy(DiscreteEntropyConfig())
        filter_fields = {
            f.name for f in dataclasses.fields(strategy.filter_criteria_type)
        }
        data_fields = {f.name for f in dataclasses.fields(strategy.data_input_type)}

        self.assertEqual(filter_fields, data_fields)


class TestIntegrationCrossEntropyStrategy(unittest.TestCase):
    """Integration tests for CrossEntropyStrategy against infomeasure estimators."""

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_cross_entropy_of_identical_distributions_equals_entropy(self) -> None:
        """H(P, P) = H(P): cross-entropy equals entropy for identical distributions."""
        data = np.tile([0.0, 1.0], 50)
        sig = AnalogSignal(data * pq.mV, sampling_rate=1.0 * pq.kHz)

        result = CrossEntropyStrategy(DiscreteEntropyConfig()).run_analysis(
            CrossEntropyStrategyDataInput(data_p=[sig], data_q=[sig])
        )

        self.assertAlmostEqual(result.matrix[0].value, np.log(2), places=10)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_cross_entropy_is_not_less_than_entropy(self) -> None:
        """H(P, Q) >= H(P) for any Q (Gibbs' inequality)."""
        data_p = np.tile([0.0, 1.0], 50)
        data_q = np.tile([0.0, 0.0, 0.0, 1.0], 25)
        sig_p = AnalogSignal(data_p * pq.mV, sampling_rate=1.0 * pq.kHz)
        sig_q = AnalogSignal(data_q * pq.mV, sampling_rate=1.0 * pq.kHz)

        cross_result = CrossEntropyStrategy(DiscreteEntropyConfig()).run_analysis(
            CrossEntropyStrategyDataInput(data_p=[sig_p], data_q=[sig_q])
        )
        entropy_result = CrossEntropyStrategy(DiscreteEntropyConfig()).run_analysis(
            CrossEntropyStrategyDataInput(data_p=[sig_p], data_q=[sig_p])
        )

        self.assertGreaterEqual(
            cross_result.matrix[0].value, entropy_result.matrix[0].value
        )
